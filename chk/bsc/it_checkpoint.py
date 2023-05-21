import torch
import os
import sys
import re
import logging
from os.path import isfile
import copy
import threading
import time
import enum
import copy
import torchvision.models as models
from collections import OrderedDict
from collections.abc import Mapping
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from torch.nn import Module
import bsc.global_value as glo
import pmemop
try:
    set_start_method('spawn')
except RuntimeError:
    pass

class ITcheckpoint:
    def __init__(self, model, optimizer, chk_way=0, use_thread=True):
        self._logger = logging.getLogger("IterCheckpoint")
        self.chk_process = None
        self.tracking_map = OrderedDict()
        self.tracking_map['model'] = model
        self.tracking_map['optimizer'] = optimizer
        self._chk_way = chk_way
        self._snapshot = OrderedDict()
        self.model_latest_snapshot = None
        self.optimizer_latest_snapshot = None
        self.additional_state = None
        self.cpu_events = {}
        for name, param in model.named_parameters():
            self.cpu_events[name] = torch.cuda.Event()
        self.pm_handles = {}
        self.param_mappings = {}
        pmemop.mmap_pmem()
        self.cpu_final_event = None
        self.active_snapshot = Value('i', 0)
        self.in_progress_snapshot = Value('i', 0)
        self.lock = Lock()
        self.use_thread = use_thread
        self.chk_global_id = 0
        self.chk_prefix = 'chk_async'
        
        self.snapshot_count = 0
        self.chk_at_step = 0
        self.chk_freq = int(os.environ.get('CHK_FREQ', '10'))
        self.steps_since_chk = 0
        self.chk_count = 0
        self.snap_start_time = 0
        torch.cuda.synchronize()
        self._logger.info("Iteration-level checkpoint initializes")
        
    def init_model_state(self):
        if self._chk_way == 0:      # snapshot to gpu
            self.model_latest_snapshot = copy.deepcopy(self.tracking_map['model'].state_dict())
            self.optimizer_latest_snapshot = copy.deepcopy(self.tracking_map['optimizer'].state_dict())
        elif self._chk_way == 1:    # snapshot to cpu
            self.model_latest_snapshot = _to_cpu(self.tracking_map['model'].state_dict())        
            self.optimizer_latest_snapshot = _to_cpu(self.tracking_map['optimizer'].state_dict())
            # check_is_pinned(self.model_latest_snapshot)
        elif self._chk_way == 2:    # snapshot to pm
            self.model_latest_snapshot = copy.deepcopy(self.tracking_map['model'].state_dict())
            self.optimizer_latest_snapshot = copy.deepcopy(self.tracking_map['optimizer'].state_dict())
            torch.cuda.synchronize()
        self._snapshot['model'] = self.model_latest_snapshot
        self._snapshot['optimizer'] = self.optimizer_latest_snapshot
        self.additional_state = glo.get_global_value()
        for key, value in self.additional_state.items():
            self._snapshot[key] = value
        
        # f = open('init_state.txt', 'w');
        # print("optimizer state dict", file=f)
        # print(self.optimizer_latest_snapshot, file=f)
        # print("model state dict", file=f)
        # print(self.model_latest_snapshot, file=f)
        # f.close()

        param_id_mapping = {}
        i = 0
        # print("param mapping")
        for name, param in self.tracking_map['model'].named_parameters():
            param_id_mapping[name]  = i
            # print(name, id(param), i)
            i += 1
        # self.snapshot_opt_param_groups(self.tracking_map['optimizer'])
        # print(self.param_mappings)
        # print("buffer")
        # self.snapshot_buffer_pm();
        # for name, v in self.buffer_dict.items():
        #     print(name)
        buffer_name_list = self.get_buffer_name_list()
        print(buffer_name_list)
        pmemop.init_model_state(self._snapshot, buffer_name_list, param_id_mapping, self._chk_way)
        torch.cuda.synchronize()
        self._logger.info("finish init model state")

    def snapshot(self, name, p):
        if self.active_snapshot.value == 0:
            with self.lock:
                self.active_snapshot.value = 1
        if self.snapshot_count == 0:
            self.snap_start_time = time.time()
            self.additional_state = glo.get_global_value()
            self.snapshot_buffer_gpu()
            self.snapshot_opt_param_groups(self.tracking_map['optimizer'])
            self._logger.info("additional state is {}".format(self.additional_state))
        # self.model_latest_snapshot[name] = copy.deepcopy(p.detach())
        self.model_latest_snapshot[name] = p.detach().clone()
        opt_state_copy = self.optimizer_latest_snapshot['state']
        opt_state = self.tracking_map['optimizer'].state
        # opt_state_copy[self.param_mappings[id(p)]] = copy.deepcopy(opt_state[p])
        _copy_dict(opt_state[p], opt_state_copy[self.param_mappings[id(p)]])
        return True

    def snapshot_cpu(self, name, p):
        if self.active_snapshot.value == 0:
            with self.lock:
                self.active_snapshot.value = 1
        if self.in_progress_snapshot.value == 0:
            with self.lock:
                self.in_progress_snapshot.value = 1
        if self.snapshot_count == 0:
            self.snap_start_time = time.time()
            self.additional_state = glo.get_global_value()
            self.snapshot_buffer_cpu()
            self.snapshot_opt_param_groups(self.tracking_map['optimizer'])
        s = time.time()
        _copy_to_cpu(p, self.model_latest_snapshot[name])
        opt_state_copy = self.optimizer_latest_snapshot['state']
        opt_state = self.tracking_map['optimizer'].state
        # _copy_to_cpu(opt_state[p], opt_state_copy[self.param_mappings[id(p)]])
        _copy_dict(opt_state[p], opt_state_copy[self.param_mappings[id(p)]])
        self.cpu_events[name].record()
        dur = time.time() - s
        # self._logger.info("Time for coping {} to cpu is {}s".format(name, dur))
        return True
    
    # directly write layer-wise parameter from gpu to pm
    def snapshot_pm(self, name, p):
        if self.in_progress_snapshot.value == 0:
            with self.lock:
                self.in_progress_snapshot.value == 1
        if self.snapshot_count == 0:
            self.snap_start_time = time.time()
            pmemop.use_new_dict()
            self.additional_state = glo.get_global_value()
            self._logger.info("additional state is {}".format(self.additional_state))
            for key, value in self.additional_state.items():
                pmemop.set_additional_item(key, value)
            self._logger.debug('finish snapshot additional_item on pm')
            self.snapshot_buffer_pm()
            self._logger.debug('finish snapshot buffer on pm')
            self.snapshot_opt_param_groups(self.tracking_map['optimizer'])
            pmemop.set_opt_param_groups(self.optimizer_latest_snapshot['param_groups'])
            self._logger.debug('finish snapshot optimizer param_groups on pm')
        opt_state = self.tracking_map['optimizer'].state
        pmemop.set_opt_state(self.param_mappings[id(p)], opt_state[p])
        self._logger.debug('finish snapshot optimizer state on pm')
        self.pm_handles[name] = pmemop.set_model_item(name, p)
        return True

    def get_buffer_name_list(self):
        buffer_name_list = []
        model = self.tracking_map['model']
        memo = set()
        modules = model.named_modules()
        for module_prefix, module in modules:
            members = module._buffers.items()
            for k, v in members:
                if v is None or v in memo or k in module._non_persistent_buffers_set:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                buffer_name_list.append(name)
        return buffer_name_list
    
    def snapshot_buffer_gpu(self):
        # self.model_latest_snapshot._metada = OrderedDict()
        model = self.tracking_map['model']
        memo = set()
        modules = model.named_modules()
        for module_prefix, module in modules:
            members = module._buffers.items()
            for k, v in members:
                if v is None or v in memo or k in module._non_persistent_buffers_set:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                self.model_latest_snapshot[name] = copy.deepcopy(v.detach())

    def snapshot_buffer_cpu(self):
        # self.model_latest_snapshot._metada = OrderedDict()
        model = self.tracking_map['model']
        memo = set()
        modules = model.named_modules()
        for module_prefix, module in modules:
            members = module._buffers.items()
            for k, v in members:
                if v is None or v in memo or k in module._non_persistent_buffers_set:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                _copy_to_cpu(v, self.model_latest_snapshot[name])
                
    def snapshot_buffer_pm(self):
        # self.model_latest_snapshot._metada = OrderedDict()
        self.buffer_dict = {}
        model = self.tracking_map['model']
        memo = set()
        modules = model.named_modules()
        for module_prefix, module in modules:
            members = module._buffers.items()
            for k, v in members:
                if v is None or v in memo or k in module._non_persistent_buffers_set:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                # print('save model buffer {}'.format(name))
                # for name, v in model.named_buffers():
                # self.pm_handles[name] = pmemop.set_model_item(name, v)
                self.buffer_dict[name] = copy.deepcopy(v.detach())
        for name, v in self.buffer_dict.items():
            self.pm_handles[name] = pmemop.set_model_item(name, v)
    
    # snapshot is on GPU, write snapshot from GPU to PM
    def save_pm(self):
        # with self.lock:
        #     self.active_snapshot.value = 1
        s = time.time()
        self._snapshot['model'] = self.model_latest_snapshot
        self._snapshot['optimizer'] = self.optimizer_latest_snapshot
        for key, value in self.additional_state.items():
            self._snapshot[key] = value
        pmemop.save_dict(self._snapshot)
        with self.lock:
            self.active_snapshot.value = 0
        self._logger.info('save_pm time {}'.format(time.time() - s))
    
    def save_cpu_pm(self):
        self._logger.debug("call save_cpu_pm")
        self._snapshot['model'] = self.model_latest_snapshot
        self._snapshot['optimizer'] = self.optimizer_latest_snapshot
        for key, value in self.additional_state.items():
            self._snapshot[key] = value
        s = time.time()
        self.cpu_final_event.synchronize()
        dur = time.time() - s
        with self.lock:
            self.in_progress_snapshot.value = 0
        self._logger.info("cpu copy synchronize time {}".format(dur))
        pmemop.save_dict(self._snapshot)
        with self.lock:
            self.active_snapshot.value = 0
            
    def save_gpu_pm(self):
        with self.lock:
            self.active_snapshot.value = 0    
    
    def snapshot_opt_param_groups(self, optimizer):
        start_index = 0
        
        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            self.param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in self.param_mappings})
            packed['params'] = [self.param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in optimizer.param_groups]
        self.optimizer_latest_snapshot['param_groups'] = param_groups


def _to_cpu(ele, snapshot=None):
    if torch.is_tensor(ele):
        snapshot = ele.cpu().pin_memory()    # return a copy of this object in cpu pinned memory
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v, snapshot[idx])
    else:
        snapshot = copy.deepcopy(ele)
    return snapshot

def _copy_to_cpu(ele, snapshot):
    if torch.is_tensor(ele):
        # check_is_pinned(snapshot)
        snapshot.copy_(ele.detach(), non_blocking=True)
    elif isinstance(ele, dict):
        for k, v in ele.items():
            _copy_to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        for idx, v in enumerate(ele):
            _copy_to_cpu(v, snapshot[idx])
    

def _copy_dict(ele, snapshot):
    for k, v in ele.items():
        if isinstance(v, dict):
            _copy_dict(v, snapshot[k])
        elif torch.is_tensor(v):
            snapshot[k].copy_(v.detach(), non_blocking=True)
        else:
            snapshot[k] = v 

def check_is_pinned(ele):
    if hasattr(ele, 'cpu'):
        if ele.is_pinned():
            print('tensor is in pinned memory')
        else:
            print('tensor not in pinned memory')
    elif isinstance(ele, dict):
        for k, v in ele.items():
            check_is_pinned(v)
    elif isinstance(ele, list):
        for v in ele:
            check_is_pinned(v)
