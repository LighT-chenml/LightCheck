## Evaluation Envirionment
We use 3 servers for evaluation, the detailed information of the server:
* 2 \* 26-core Intel Xeon Gold 6230R CPUs, 1 Tesla V100 GPU
* 6 \* 32GB DDR4 DIMMs, 6 \* 128GB Intel Optane DC DIMMs, 3.6TB HDD
* 1 100Gbps Mellanox ConnectX-5 InfiniBand RNIC
* Ubuntu 18.04 with Linux kernel version 5.4.0
* Python 3.6.9
* CUDA toolkit 11.1 with PyTorch 1.8.1
* Horovod

## Setup
```
cd pmemtest
python setup.py install
```

## Example
```
export RUN_SCHEDULER=1
export RUN_CHECKPOINT=1
export CHK_WAY=0
horovodrun -np 3 -H ip1:1,ip2:1,ip3:1 -p 12262 python dali_bench.py --noeval -j 8 -b 128 -a resnet18 --dali --epochs 2
```
