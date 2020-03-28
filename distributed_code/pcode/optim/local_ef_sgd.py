# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
import torch.distributed as dist
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
                local_scale, local_sign = [], []
                local_compressed = []
                for consensus_param, param, memory in zip(
                    self.consensus_params_tb, params_tb, self.memory_tb
                ):
                    # add memory to the model difference.
                    memory.data.copy_(consensus_param - param + memory)
                    local_compressed.append(memory.data.clone())
                    # compress.
                for consensus_param, param, memory in zip(
                    self.consensus_params_tb, params_tb, self.memory_tb
                ):                    
                    _local_scale, _local_sign = scaled_sign(memory)
                    # update memory.
                    memory.data.copy_(memory - _local_scale * _local_sign)
                    # store local scales and local sign.
                    local_sign.append(_local_sign)
                    local_scale.append(_local_scale)

                # concat the update magnitude and directions.
                magnitudes_tb = TensorBuffer(local_scale)
                directions_tb = TensorBuffer(local_sign)
                compressed_tb = TensorBuffer(local_compressed)
                compressed, padding = quantize_gpu(compressed_tb.buffer, 1)
                print(unquantize_gpu(compressed, padding, 1) - directions_tb.buffer)
            # sync and decompress.
            with kargs["timer"]("sync/sync_and_decompress", epoch=self.conf.epoch_):
                # sync the directions.
                compressed = self.world_aggregator._agg(
                    compressed, "avg", distributed=self.conf.distributed
                )
                rank = dist.get_rank()
                chunks = list(compressed_tb.buffer.view(dist.get_world_size(), -1))
                for i, chunk in enumerate(chunks):
                    small_chunk, padding = quantize_gpu(chunk, 1)
                    #dist.g(small_chunk, i, op=dist.ReduceOp.SUM)
                    small_chunks = []*2
                    dist.gather(small_chunk, i, small_chunks)
                    if i == rank:
                        decompressed = list(map(lambda tensor: unquantize_gpu(tensor, padding, 1), small_chunks))
                        chunks[rank] = torch.stack(decompressed).sum()
                chunk = chunks[rank]
                dist.all_gather(chunks, chunk)
                directions_tb.buffer = self.world_aggregator._agg(
                    directions_tb.buffer, "avg", distributed=self.conf.distributed
                )
                magnitudes_tb.buffer = self.world_aggregator._agg(
                    magnitudes_tb.buffer, "avg", distributed=self.conf.distributed
                )
            #compressed_tb.buffer #= unquantize_gpu(compressed, padding, 1)
            print(compressed_tb.buffer - directions_tb.buffer)
            # unpack the synced info and update the consensus params.
            with kargs["timer"]("sync/update_consensus", epoch=self.conf.epoch_):
                for update_magnitude, update_direction, consensus_param in zip(
                    magnitudes_tb, compressed_tb, self.consensus_params_tb
                ):
                    consensus_param.add_(-1.0, update_direction.mul(update_magnitude))

            # consistent the local models by assigning the consensus params.
            self.consensus_params_tb.unpack(params)
            n_bits = get_n_bits(directions_tb.buffer) + get_n_bits(magnitudes_tb.buffer)
        else:
            n_bits = 0
        return n_bits

def scaled_sign(x, name=None):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm divided by the number of elements
    """
    _scale = x.norm(p=1) / x.numel()
    _sign = torch.sign(x)

    return _scale, _sign