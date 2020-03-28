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

def send(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	dist.broadcast(tensor, rank, group=private)

def recv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(tensor, src, group=private)

def allreduce(tensor):
    rank = dist.get_rank()
    N = dist.get_world_size()
    chunks = list(tensor.view(N, -1))
    peers = list(filter(lambda r: not r == rank, range(N)))
    print('send')
    padding = 0
    pad_size = list(chunks[0].size())[0] % 32
    padding = (32 - pad_size) % 32
    compressed_chunks = []*N
    for i in peers:
        compressed_chunk = quantize_gpu(chunks[i], 1)
        send(compressed_chunk, i)
        compressed_chunks[i] = compressed_chunk
    print('recv')
    buf = torch.zeros(chunks[0].size())
    for i in peers:
        recv(buf, i)
        chunks[rank] = unquantize_gpu(buf, padding, 1)
    print('all_gather')
    compressed_chunks[rank], padding = quantize_gpu(chunks[rank], 1)
    dist.all_gather(compressed_chunks, compressed_chunks[rank])
    for i, chunk in enumerate(chunks):
        chunk.data[:] = unquantize_gpu(compressed_chunks[i], padding, 1)

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
            # sync and decompress.
            with kargs["timer"]("sync/sync_and_decompress", epoch=self.conf.epoch_):
                # sync the directions.
                allreduce(compressed_tb.buffer)
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