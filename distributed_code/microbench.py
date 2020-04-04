from copy import deepcopy

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer
from lib import quantize_gpu, unquantize_gpu, CUDATimer
import argparse
import os
import datetime

def send(tensor, dst):
	rank = dist.get_rank()
	#private = dist.new_group([rank, dst])
	dist.broadcast(tensor, rank)

def recv(tensor, src):
	#private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(tensor, src)

def allreduce(tensor):
    rank = dist.get_rank()
    N = dist.get_world_size()
    chunks = list(tensor.view(N, -1))
    compressed_chunks = [None]*N
    chunks[rank][:] = torch.sign(chunks[rank])
    compressed_chunk, padding = quantize_gpu(chunks[(rank+1)%2], 1)
    compressed_chunks[(rank+1)%2] = compressed_chunk
    buf = torch.zeros(compressed_chunk.size(), device=tensor.device)
    if rank == 0:
        send(compressed_chunk, 1)
        recv(buf, 1)
        chunks[rank] += unquantize_gpu(buf, padding, 1)

    elif rank == 1:
        recv(buf, 0)
        chunks[rank] += unquantize_gpu(buf, padding, 1)
        send(compressed_chunk, 0)    
    compressed_chunks[rank], padding = quantize_gpu(chunks[rank], 1)
    dist.all_gather(compressed_chunks, compressed_chunks[rank])
    chunks[(rank+1)%2] = unquantize_gpu(compressed_chunks[(rank+1)%2], padding, 1)
    chunks[rank] /= N
    
    tensor[:] = torch.stack(chunks).view(tensor.size())

def centralized_allreduce(tensor, timer):
    rank = dist.get_rank()
    N = dist.get_world_size()
    with timer('compress-1'):
        to_send, padding = quantize_gpu(tensor, 1)
    with timer('clone'):
        buf = to_send.clone()
    with timer('exchange'):
        if rank == 0:
            recv(buf, 1)
            send(to_send, 1)
        else:
            send(to_send, 0)
            recv(buf, 0)
    with timer('decompress'):
        recv_ed = unquantize_gpu(buf, padding, 1)
    s = torch.sign(tensor)
    tensor[:] = (recv_ed + s) / 2

def scaled_sign(x, name=None):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm divided by the number of elements
    """
    _scale = x.norm(p=1) / x.numel()
    _sign = torch.sign(x)

    return _scale, _sign

def benchmark1(tensors):
    timer = CUDATimer('baseline')
    local_scale, local_sign = [], []
    with timer('compression'):
        for tensor in tensors:
            _local_scale, _local_sign = scaled_sign(tensor)
            local_scale.append(_local_scale)
            local_sign.append(_local_sign)
    with timer('flattening'):
        magnitudes_tb = TensorBuffer(local_scale)
        directions_tb = TensorBuffer(local_sign)
    with timer('com'):
        dist.all_reduce(magnitudes_tb.buffer, op=dist.ReduceOp.SUM)
        magnitudes_tb.buffer /= 2
        dist.all_reduce(directions_tb.buffer, op=dist.ReduceOp.SUM)
        directions_tb.buffer /= 2
    timer.upload_raw('microbenchmarking', 
        {
            'microbenchmark': 'sign_sgd_com', 
            'input': list(map(lambda t: t.size(), tensors))
        }
    )

def benchmark2(tensors):
    timer = CUDATimer('centralized_allreduce')
    local_compressed, local_scale = [], []
    with timer('compression'):
        for tensor in tensors:
            local_compressed.append(tensor.clone())
        for tensor in tensors:
            _local_scale, _local_sign = scaled_sign(tensor)
            # store local scales and local sign.
            local_scale.append(_local_scale)
    with timer('flattening'):
        magnitudes_tb = TensorBuffer(local_scale)
        #directions_tb = TensorBuffer(local_sign)
        compressed_tb = TensorBuffer(local_compressed)
    with timer('com'):
        centralized_allreduce(compressed_tb.buffer, timer)
                    #print('difff after', compressed_tb.buffer - directions_tb.buffer)
        dist.all_reduce(magnitudes_tb.buffer, op=dist.ReduceOp.SUM)
        magnitudes_tb.buffer /= 2
    timer.upload_raw('microbenchmarking', 
        {
            'microbenchmark': 'sign_sgd_com', 
            'input': list(map(lambda t: t.size(), tensors))
        }
    )
    #timer.dump()

def rendezvous(backend, rank, world_size):
    dist.init_process_group(backend, rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    rendezvous('nccl', rank, 2)
    conf = object()
    for size in range(10, 24):
        tensors = [torch.ones(2**size).cuda() for i in range(10)]
        benchmark1(tensors)
        tensors = [torch.ones(2**size).cuda() for i in range(10)]
        benchmark2(tensors)