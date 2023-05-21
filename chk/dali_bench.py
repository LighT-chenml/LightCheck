import time
import argparse
import os
import random
import shutil
import warnings
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import horovod.torch as hvd
from collections import OrderedDict
import bsc.global_value as glo
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import threading

glo.init()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="/home/data/tiny-imagenet-200/", type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--noeval', action='store_true', help = 'not run evaluation phase')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for initializing training')
parser.add_argument('--download-data', action='store_true', default=False,
                    help='download data for training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument("--classes", default=1000, type=int)

parser.add_argument("--cache_size", default=0, type=int)
parser.add_argument('--dali', action='store_true')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        # self.input = ops.FileReader(file_root=data_dir, shard_id=hvd.rank(), num_shards=hvd.size(), shuffle_after_epoch=True)
        self.input = ops.readers.File(file_root=data_dir, shard_id=hvd.rank(), num_shards=hvd.size(), shuffle_after_epoch=True)

        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        #decoder_device = 'cpu' 
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.decoders.ImageRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            # output_dtype=types.FLOAT,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            # image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        # self.coin = ops.CoinFlip(probability=0.5)
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # shard = int(args.node_rank*args.world_size/args.nnodes + args.local_rank)
        # self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, random_shuffle=False)
        # self.input = ops.FileReader(file_root=data_dir, shard_id=hvd.rank(), num_shards=hvd.size(), random_shuffle=False)
        self.input = ops.readers.File(file_root=data_dir, shard_id=hvd.rank(), num_shards=hvd.size(), random_shuffle=False)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="cpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            # output_dtype=types.FLOAT,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            # image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]

best_acc1 = 0
args = parser.parse_args()

if args.dali:
    print("Using DALI")
else:
    print("Using native dataloader")

def main():
    global best_acc1, args
   
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)
    
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    start_full = time.time()
    
    time_stat = OrderedDict()
    start = time.time()
    
    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(8)
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if(args.arch == "inception_v3"):
            model = models.__dict__[args.arch](num_classes=args.classes,aux_logits=False)
        else:
            model = models.__dict__[args.arch]()
        
    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batch_size * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    optimizer = torch.optim.SGD(model.parameters(), lr=(args.lr * lr_scaler),
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    
    # bytescheduler wrapper
    use_scheduler = int(os.environ.get('RUN_SCHEDULER', '0'))
    if use_scheduler > 0:
        import bsc.bsc_horovod as bsc
        bsc.init()
    
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        # backward_passes_per_step=args.batches_per_allreduce,
        backward_passes_per_step=1,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    
    # optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    if args.resume and hvd.rank() == 0:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # Horovod: broadcast start_epoch from rank 0 to other ranks
            args.start_epoch = hvd.broadcast(torch.tensor(args.start_epoch), root_rank=0,
                                             name='start_epoch').item()
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_pipe = None
    if args.dali:
        if(args.arch == "inception_v3"):
            crop_size = 299
            val_size = 320 # I chose this value arbitrarily, we can adjust.
        else:
            crop_size = 224
            val_size = 256

        pipe = HybridTrainPipe(batch_size=args.allreduce_batch_size, num_threads=args.workers, device_id=hvd.local_rank(), data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
        
        pipe.build()
        train_pipe = pipe

        # train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size), fill_last_batch=False, resume_size=resume_size)
        train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / hvd.size()), fill_last_batch=False)
        # train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / hvd.size()), last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True)

        if not args.noeval:
            pipe_val = HybridValPipe(batch_size=args.allreduce_batch_size, num_threads=args.workers, device_id=hvd.local_rank(), data_dir=valdir, crop=crop_size, size=val_size)
            pipe_val.build()
            val_loader = DALIClassificationIterator(pipe_val, size=int(pipe_val.epoch_size("Reader") / hvd.size()))

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print('use tiny-imagenet dataset')
    
        # Horovod: use DistributedSampler to partition data among workers. Manually specify
        # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        print('hvd size {}, rank {}, train dataset length {}'.format(hvd.size(), hvd.rank(), len(train_dataset)))
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.allreduce_batch_size,
            # num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            # num_workers=args.workers, pin_memory=True, prefetch_factor=128, sampler=train_sampler)
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.allreduce_batch_size,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    if use_scheduler > 0:
        if args.dali:
            train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
        else:
            train_loader_len = int(len(train_loader))
        optimizer = bsc.ScheduledOptimizer(model, optimizer)

    if args.evaluate:
        validate(val_loader, model, args)
        return

    dur_setup = time.time() - start
    time_stat["setup_time"] = dur_setup
    train_ep = AverageMeter('Train Time', ':6.3f')
    
    # enable_profiling = args.profiler & (hvd.rank() == 0)
    # with torch.autograd.profiler.profile(enabled=enable_profiling, use_cuda=True) as prof:
    for epoch in range(args.start_epoch, args.epochs):
        print("epoch {}".format(epoch))
        start_ep = time.time()
        if not args.dali:
            train_sampler.set_epoch(epoch)
        
        glo.set_value('epoch', epoch)

        # train for one epoch
        avg_train_time = train(train_loader, model, criterion, optimizer, epoch, args)
        train_ep.update(avg_train_time)
        
        # evaluate on validation set
        if args.noeval:
            acc1 = 0
        else:
            acc1 = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if hvd.rank() == 0:
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_acc1': best_acc1,
            #     'optimizer' : optimizer.state_dict(),
            # }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                    '##Perf  {1}'.format(acc1, hvd.size() * args.allreduce_batch_size / train_ep.avg))
        
        dur_ep = time.time() - start_ep
        print("epoch {} takes {}s".format(epoch, dur_ep))
        time_stat["epoch" + str(epoch)] = dur_ep
        
        if args.dali:
            # reset DALI iterators
            train_loader.reset()
            if not args.noeval:
                val_loader.reset()

    # if enable_profiling:
    #     print("export profiling")
    #     prof.export_chrome_trace(os.path.join('pytorch-trace', args.arch+'-'+str(hvd.rank()) +'.json'))
    
    dur_full = time.time() - start_full
    if hvd.rank() == 0:
        with open("time.txt", 'w') as f:
            for k, t in time_stat.items():
                print("Time stat {} : {}s".format(k, t))
                f.write(str(t))
                f.write("\n")
        print("Total time for all {} epochs = {}s".format(args.epochs-args.start_epoch, dur_full))
    
    if use_scheduler > 0:
        optimizer.bsc_exit()
        
    if args.dali:
        del pipe
        if not args.noeval:
            del pipe_val 


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if args.dali:
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    else:
        train_loader_len = int(len(train_loader))
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    def trace_handler(prof):
        print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("pytorch-trace/pytorch-trace" + str(prof.step_num) + ".json")
    
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=20),
    #     on_trace_ready=trace_handler
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     # used when outputting for tensorboard
    # ) as p:
    if True:
    # event1 = torch.cuda.Event()
        all_iterations = train_loader_len
        # for i, (images, target) in enumerate(train_loader):
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            adjust_learning_rate(optimizer, train_loader, epoch, i, args, train_loader_len)
            
            if args.dali:
                input_var = data[0]["data"]
                target_var = data[0]["label"].squeeze().cuda().long()
                images = Variable(input_var)
                target = Variable(target_var)
            else:
                input_var, target_var = data
                target_var = target_var.squeeze().cuda().long()
                images = Variable(input_var).cuda(non_blocking=True)
                target = Variable(target_var).cuda(non_blocking=True)
            
            glo.set_value('iter_this_epoch', i)
            data_index = i * args.allreduce_batch_size
            glo.set_value('data_index', data_index)
            
            # compute output
            k = 0
            for j in range(0, len(images), args.batch_size):
                # s2 = time.time()
                optimizer.zero_grad()
                images_batch = images[j:j + args.batch_size]
                target_batch = target[j:j + args.batch_size]
                output = model(images_batch)
                # s3 = time.time()
                # print('forward time {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                # loss = F.cross_entropy(output, target_batch)
                loss = criterion(output, target_batch)
                acc1, acc5 = accuracy(output, target_batch, topk=(1, 5))
                # loss_val = loss.detach().to(device='cpu', non_blocking=True)
                loss_val = loss.data
                # loss_val = loss.item()
                # torch.cuda.current_stream().synchronize()
                # event1.record()
                # event1.synchronize()
                # s3 = time.time()
                # print('get loss in cpu {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                # losses.update(loss.item(), images_batch.size(0))
                losses.update(loss_val, images_batch.size(0))
                # s3 = time.time()
                # print('upate losses {}, {}'.format(s3 - s2, s3 - end))
                # s2 = time.time()
                top1.update(acc1[0], images_batch.size(0))
                top5.update(acc5[0], images_batch.size(0))
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(images)) / args.batch_size))
                loss.backward()
                # s3 = time.time()
                # print('backward time {}, {}'.format(s3 - s2, s3 - end))
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                # if i % args.print_freq == 0:
                print('iter {} of {}, batch time: {}'.format(i * args.batches_per_allreduce + k, all_iterations, time.time() - end))
                end = time.time()
                k = k + 1
                
            # if i % args.print_freq == 0:
                # progress.display(i * args.batches_per_allreduce + k)
            
                # p.step()
    progress.display(all_iterations)
    return batch_time.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.cuda:
                images, target = images.cuda(), target.cuda()
            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, train_loader, epoch, batch_idx, args, train_loader_len):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / train_loader_len
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * args.batches_per_allreduce * lr_adj

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
