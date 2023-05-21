from __future__ import absolute_import
import os
import threading
import logging
import time
import copy
try:
    import queue
except ImportError:
    import Queue as queue
import torch
import horovod.torch as hvd
import horovod.torch.optimizer as hopt
from horovod.torch.mpi_ops import size, rank, broadcast_async, synchronize
import time
import math
from bsc.it_checkpoint import ITcheckpoint
import bsc.global_value as glo
import pmemop
from collections import OrderedDict
from torch.multiprocessing import Pool, Process, set_start_method
# import bsc.utils as utils
try:
    set_start_method('spawn')
except RuntimeError:
    pass

class ScheduledOptimizer(hopt._DistributedOptimizer):
    def __init__(self, model, hvd_opt):
        self._model = model
        self._opt = hvd_opt
        self._logger = logging.getLogger("IterCheckpoint")
        self._logger.info("hvd size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())
        
        # Track training steps
        self._step = 0
        self._update_time = 0
        self._allreduce_time = 0
        self._lock = threading.Lock()
        
        # Use lock to block the forward propagation of each parameter.
        self._locks = {}
        # self._first_param_name, first_param = next(self._model.named_parameters())
        self._num_param_layers = len(self._parameter_names)
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()
        
        assert len(self._grad_accs) == 0
        # self._register_forward_hooks()
        self._register_hooks()
        
        # Poll whether the allreduce is finished.
        self.event_queue = queue.Queue()
        self._poller = threading.Thread(target=self._poll, args=())
        self._poller.start()
        self._start_update = None

        # server_pid = glo.get_value("server_pid")
        # if server_pid != None:
        #     print("set mps percentage")
        #     utils.mps_set_active_thread_percentage(server_pid, "10")
        #     time.sleep(4)
        
        # Let rank 0 do checkpoint
        self._rank = rank()
        run_checkpoint = int(os.environ.get('RUN_CHECKPOINT', '0'))
        if self._rank == 0 and run_checkpoint != 0:
            self._run_checkpoint = 1
        else:
            self._run_checkpoint = 0
        if self._run_checkpoint != 0:
            self._chk_way = int(os.environ.get('CHK_WAY', '0'))
            self.ITchk = ITcheckpoint(self._model, self._opt, self._chk_way)
            self.checkpoint_queue = queue.Queue()
            self._checkpointer = threading.Thread(target=self._handle_chk, args=())
            self._checkpointer.start()
    
    def __getattr__(self, item):
        return getattr(self._opt, item)

    #def __del__(self):
    def bsc_exit(self):
        """Clean up"""
        self.event_queue.put((None, None, None, None))
        self._poller.join()
        if self._run_checkpoint != 0:
            pmemop.unmmap_pmem()
            self.checkpoint_queue.put((None, None))
            self._checkpointer.join()
        self._logger.info("bsc horovod clean")
    
    def step(self, closure=None):
        """Override the default step function."""
        self._logger.debug("{} calls step() {}".format(self._desc, self._step))
        
        # Step 0 is called for parameter initialization after parameter broadcast
        if self._step > 0:
            loss = None
            if closure is not None:
                loss = closure()
            with self._lock:
                self._step += 1
                self._logger.debug('{} finish {} step update'.format(self._desc, self._step - 1))
            return loss
        else:
            # SGD.step() will be triggered when user calls hvd.broadcast_optimizer_sate()
            super(self._opt.__class__, self._opt).step()
            self._step += 1
    
    def zero_grad(self):
        """Override the default zero_grad function

        Clears the gradients of all optimized :class:`torch.Tensor` s.
        """
        self._logger.debug("{} calls zero_grad() of step {}".format(self._desc, self._step))
        if self._step > 0:
            return
        else:
            self._opt.zero_grad()

    def _poll(self):
        """Poll the completion of the tensor's allreduce from a FIFO event_queue"""
        while True:
            p, grad, handle, ctx = self.event_queue.get()
            if p is None:
                self._logger.debug("poller exits.")
                break
            
            # Check whether the allreduce is finished. If so, start updating parameters.
            if handle is not None and hvd.mpi_ops.poll(handle):
                output = hvd.mpi_ops.synchronize(handle)
                grad.set_(self._compression.decompress(output, ctx))
                self._allreduce_delay[p] = self.backward_passes_per_step
                param_name = self._parameter_names[p]
                if self._run_checkpoint != 0:
                    if self.ITchk.steps_since_chk == 1 and self._chk_way == 1 and self._step > 0:
                        self._logger.debug('update times {}, in_progress_snapshot {}, time {}'.format(self._update_time, self.ITchk.in_progress_snapshot.value, time.time()))
                        if self._update_time == 0 and self.ITchk.in_progress_snapshot.value == 1:
                            self._logger.info('whether last batch checkpoint finish {}'.format(self.ITchk.in_progress_snapshot.value))
                        while self._update_time == 0 and self.ITchk.in_progress_snapshot.value == 1:
                            continue
                        if self.ITchk.cpu_events[param_name].query() == False:
                            self._logger.info('last batch checkpoint is not finished')
                            self.ITchk.cpu_events[param_name].synchronize()
                    elif self.ITchk.steps_since_chk == 1 and self._chk_way == 2 and self._step > 4:
                        if pmemop.query_handle(self.ITchk.pm_handles[param_name]) == False:
                           self._logger.info('last batch checkpoint is not finished')
                           pmemop.wait_handle(self.ITchk.pm_handles[param_name]) 
                    # if self._update_time == 0:
                    #     torch.cuda.synchronize()
                if self._update_time == 0:
                    self._start_update = time.time()
                # So far only supports SGD optimizer and Adam optimizer
                if isinstance(self._opt, torch.optim.SGD):
                    self._sgd(p)
                elif isinstance(self._opt, torch.optim.Adam):
                    self._adam(p)
                elif isinstance(self._opt, torch.optim.AdamW):
                    self._adamw(p)
                else:
                    self._logger.error("unknown optimizer!")
                self._zero_one_grad(p)
                self._update_time +=1
                self._logger.debug('{} update times {} of {}, qsize {}'.format(self._desc, self._update_time, self._num_param_layers, self.event_queue.qsize()))
                
                # put parameter into checkpoint queue
                if self._run_checkpoint != 0:
                    if self.ITchk.steps_since_chk == self.ITchk.chk_freq:
                        self._put_chk(param_name, p)
                
                # notify sgd completion and parameter is ready for forward propagation
                if self._update_time == self._num_param_layers:
                    self._update_time = 0
                    self._logger.debug('update duration {}'.format(time.time() - self._start_update))
                    if self._run_checkpoint != 0:
                        if self._step > 2:
                            self.ITchk.steps_since_chk = self.ITchk.steps_since_chk + 1
                        if self._step == 0:
                            self._logger.info("write initial model state for checkpoint")
                            self.ITchk.init_model_state()
                        self._logger.info('steps since chk {}'.format(self.ITchk.steps_since_chk))
                    self._logger.debug('{} release lock'.format(self._desc))
                    self._lock.release()
            else:
                self.event_queue.put((p, grad, handle, ctx)) 
                
    def _put_chk(self, param_name, p):
        if self._update_time == 1 and \
            (self.ITchk.active_snapshot.value == 1 or pmemop.query_save() == False):
            self._logger.info("last checkpoint is not finished")
            self.ITchk.steps_since_chk -= 1
        else:
            self._logger.debug("put {} into chk queue".format(param_name))
            self.checkpoint_queue.put((param_name, p))
            if self._update_time == self._num_param_layers:
                self.ITchk.steps_since_chk = 0
    
    def _handle_chk(self):
        s1 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            while True:
                name, p = self.checkpoint_queue.get()
                if p is None:
                    self._logger.debug("checkpointer exits.")
                    break
                if self.ITchk.snapshot_count == 0:
                    self.ITchk.chk_at_step = self._step
                if self._chk_way == 0:
                    self.ITchk.snapshot(name, p)
                elif self._chk_way == 1:
                    self.ITchk.snapshot_cpu(name, p)
                elif self._chk_way == 2:
                    self.ITchk.snapshot_pm(name, p)
                self.ITchk.snapshot_count += 1
                if self.ITchk.snapshot_count == self._num_param_layers:
                    self.ITchk.chk_count += 1
                    dur = time.time() - self.ITchk.snap_start_time
                    if self._chk_way == 0:
                        self._logger.info('chk {}, gpu snapshot time {} of step {}'.format(self.ITchk.chk_count, dur, self.ITchk.chk_at_step))
                        self.ITchk.save_pm()
                    elif self._chk_way == 1:
                        self._logger.info('chk {}, cpu snapshot time {} of step {}'.format(self.ITchk.chk_count, dur, self.ITchk.chk_at_step))
                        self.ITchk.cpu_final_event = self.ITchk.cpu_events[name]
                        self.ITchk.save_cpu_pm()
                    elif self._chk_way == 2:
                        self._logger.info('chk {}, pm snapshot time {} of step {}'.format(self.ITchk.chk_count, dur, self.ITchk.chk_at_step))
                        self.ITchk.save_gpu_pm()
                    self.ITchk.snapshot_count = 0

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the allreduce and
        parameter update is finished. The blocking is implemented using a lock."""
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)
        
        def pre_forward_hook(mod, input):
            for p in mod.parameters():
                if p not in self._locks:
                    continue
                #with self._locks[p]:
                with self._lock:
                    self._logger.debug("{} {} is ready.".format(self._desc, self._parameter_names[p]))
                    break
            self._logger.debug("{} starts forward {}.".format(self._desc, mod))
        
        def after_forward_hook(mod, input, result):
            self._logger.debug("{} finished forward {}.".format(self._desc, mod))
            
        for mod in reversed(submodules):
            self._logger.debug("{} registers forward hook on module {}".format(self._desc, mod))
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)
            break           
    
    def _register_hooks(self):
        """Add a hook after the backward propagation of each layer to start allreduce"""
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    # p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)
    
    def _allreduce_grad_async(self, p):
        if p.grad is None:
            # Gradient was not computed, but we still need to submit a tensor to allreduce
            # as one of the other ranks may have computed it (due to dynamic forward functions).
            #
            # NOTE: this will not work if the gradient is sparse and we perform an allgather.
            # Unfrotunately, there doesn't appear to be a good way to detect that the parameter will
            # produce sparse gradients before computing the gradient.
            p.grad = p.data.new(p.size()).zero_()
        
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        handle = hvd.mpi_ops.allreduce_async_(tensor_compressed, average=True, name=name)
        return handle, ctx
    
    def _make_hook(self, p):
        """Define hook for backward propogation.
        
        Arguments:
            p: the parameter
        """
        self._logger.debug("{} calls make_hook for {}".format(self._desc, self._parameter_names[p]))
        def hook(*ignore):
            self._logger.debug("{} finished backward of {}, delay {}".format(self._desc, self._parameter_names[p],
                                                                             self._allreduce_delay[p]))
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            param_name = self._parameter_names[p]
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                # if self._first_param_name == param_name:
                #     self._locks[p].acquire()
                self._allreduce_time += 1;
                if self._allreduce_time == self._num_param_layers:
                    self._logger.debug("{} final allreduce request".format(self._desc))
                    self._allreduce_time = 0
                    self._lock.acquire()
                    self._logger.debug('{} acquire lock'.format(self._desc))
                handle, ctx = self._allreduce_grad_async(p)
                # put handle and ctx into event_queue
                self.event_queue.put((p, p.grad, handle, ctx))
        return hook

    def load_state_dict(self, *args, **kwargs):
        print("go into bsc optimizer load state dict")
        self._opt.load_state_dict(*args, **kwargs)

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as PyTorch accumulates gradients by default.

        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            # Not sure whether to do detach_ or not
            p.grad.detach_()
            p.grad.zero_()

    """Below are the implementations of optimizers, e.g., SGD, Adam."""
    
    def _sgd(self, p):
        """Performs a single optimization step using SGD optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        # TODO: support other optimizers later, or figure out a walk around way
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                #p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-group['lr'])
                break

    def _adam(self, p):
        """Performs a single optimization step using Adam optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                break
    
    def _adamw(self, p):
        for group in self.param_groups:
            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2 = group['betas']
                # update the steps for each param group update
                state['step'] += 1

                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul(grad, grad, value=1 - beta2)
                if amsgrad:
                     # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                break

def init():
    """Replace _register_hook() function in hvd._DistributedOptimizer with empty function."""

    def hijack(obj, func_name):
        orig_func = getattr(obj, func_name)
        print("hijack function {}".format(orig_func))

        def wrapped_func(*args, **kwargs):
            print("function {} is hijacked to do nothing.".format(orig_func))
            return
        setattr(obj, func_name, wrapped_func)

    hijack(hopt._DistributedOptimizer, '_register_hooks')    
