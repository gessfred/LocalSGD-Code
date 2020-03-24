# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer

from lib import quantize_gpu, unquantize_gpu, CompressedTensorBuffer


class Local_EFSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Local_EFSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.local_step = conf.local_step
        self.turn_on_local_step_from_epoch = conf.turn_on_local_step_from
        
        self.bits = conf.compress_width

        # define the aggregator.
        self.world_aggregator = comm.get_aggregators(
            conf,
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # initialize the concensus
        self._init_consensus()
        self._init_memory()

    def _init_consensus(self):
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.consensus_params_tb = deepcopy(TensorBuffer(params))

    def _init_memory(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.memory_tb = TensorBuffer(params)
        self.memory_tb.buffer = torch.zeros_like(self.memory_tb.buffer)

    def __setstate__(self, state):
        super(Local_EFSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # do the local update steps.
        with kargs["timer"]("sync/local_update", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        # enter the global sync if it satisfies the condition.
        if (
            self.conf.epoch_ < self.turn_on_local_step_from_epoch
            or self.conf.local_index % self.local_step == 0
        ):
            with kargs["timer"]("sync/get_params", epoch=self.conf.epoch_):
                # get parmas.
                params, _ = comm.get_data(
                    self.param_groups, self.param_names, is_get_grad=False
                )
                params_tb = TensorBuffer(params)
            with kargs['timer']('sync/memory_and_compress', epoch=self.conf.epoch_):
                # get the params difference w.r.t. previous synced model.
                local = []
                for consensus_param, param, memory in zip(
                    self.consensus_params_tb, params_tb, self.memory_tb
                ):
                    # add memory to the model difference.
                    memory.data.copy_(consensus_param - param + memory)
                    # compress.
                    #_local_scale, _local_sign = scaled_sign(memory)
                local_tb = CompressedTensorBuffer(local, self.conf.compress_width)
                for _local, memory in zip(
                    local_tb, self.memory_tb
                ):    
                    # update memory.
                    memory.view(-1).copy_(memory.view(-1) - unquantize_gpu(_local.view(-1), self.bits)) #very bad, just a test
                    # store local scales and local sign.
                    #local_scale.append(_local_scale)
                    #local_sign.append(_local_sign)

                # concat the update magnitude and directions.

            # sync and decompress.
            with kargs["timer"]("sync/sync_and_decompress", epoch=self.conf.epoch_):
                # sync the directions.
                local_tb.buffer = self.world_aggregator._agg(
                  local_tb.buffer, 'avg', distributed=self.conf.distributed
                )
                local_tb.decompress()

            # unpack the synced info and update the consensus params.
            with kargs["timer"]("sync/update_consensus", epoch=self.conf.epoch_):
                for update_local, consensus_param in zip(
                    local_tb, self.consensus_params_tb
                ):
                    consensus_param.add_(-1.0, update_local)

            # consistent the local models by assigning the consensus params.
            self.consensus_params_tb.unpack(params)
            n_bits = get_n_bits(local_tb.buffer) 
        else:
            n_bits = 0
        return n_bits

