# NCCL Collectives Visual Guide

A dependency-free static site that explains and visualizes NCCL collectives and
distributed-training communication patterns.

## Local preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Contents

- Interactive simulator for AllReduce, ReduceScatter, AllGather, Broadcast,
  Reduce, AllToAll, Gather, Scatter, and Barrier.
- NCCL execution model, topology, transports, channels, protocols, and group
  calls.
- Distributed-training communication patterns for DDP, FSDP/ZeRO, tensor
  parallelism, pipeline parallelism, and MoE.
- Performance and debugging sections with links to official NVIDIA and PyTorch
  documentation.
