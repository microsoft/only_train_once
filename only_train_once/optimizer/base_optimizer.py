from abc import abstractclassmethod
from torch.optim.optimizer import Optimizer, required
import torch

class BaseOptimizer(Optimizer):
    def __init__(self, params, defaults=dict(), **kwargs):
        super(BaseOptimizer, self).__init__(params, defaults)
        self.num_steps = 0
        self.safe_guard = 1e-8
        self.first_moment_grads = dict()
        self.second_moment_grads = dict()
        
    def get_second_momentum_grad_square(self, name, second_moment, dampening, grad):
        if second_moment > 0:
            if name not in self.second_moment_grads:
                buf = self.second_moment_grads[name] = grad * grad
            else:
                buf = self.second_moment_grads[name]
                buf.mul_(second_moment).add_(grad * grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad * grad

    def __setstate__(self, state):
        super(BaseOptimizer, self).__setstate__(state)

    def get_first_momentum_grad(self, name, first_moment, dampening, grad):
        if first_moment > 0:
            if name not in self.first_moment_grads:
                buf = self.first_moment_grads[name] = grad
            else:
                buf = self.first_moment_grads[name]
                buf.mul_(first_moment).add_(grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad

    def compute_grad_variant(self):
        for i, group in enumerate(self.param_groups):
            is_adam = group['variant'] == 'adam' or group['variant'] == 'adamw'
            first_bias_correction = 1.0 - group['first_momentum'] ** self.num_steps if is_adam else None
            second_bias_correction = 1.0 - group['second_momentum'] ** self.num_steps if is_adam else None
            group['grad_variant'] = dict()
            for j, (p_name, p) in enumerate(zip(group['p_names'], group['params'])):
                if p.grad is None:
                    continue
                refined_grad_f = torch.clone(p.grad.data).detach()
                if group['weight_decay'] is not None and group['variant'] != 'adamw':
                    refined_grad_f += group['weight_decay'] * p.data
                if not is_adam:
                    if group['first_momentum'] > 0.0 or group['dampening'] > 0.0:
                        refined_grad_f = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                            group['first_momentum'], group['dampening'], refined_grad_f)
                    group['grad_variant'][p_name] = refined_grad_f
                else:
                    first_moment_grad = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                        group['first_momentum'], group['first_momentum'], refined_grad_f) 
                    second_moment_grad_sq = self.get_second_momentum_grad_square(f"grad_second_moment_buffer_group_{i}_param_{j}", 
                        group['second_momentum'], group['second_momentum'], refined_grad_f)

                    exp_avg_first_moment_grad = first_moment_grad / first_bias_correction
                    exp_avg_second_moment_grad_sq = second_moment_grad_sq / second_bias_correction
                    denom = exp_avg_second_moment_grad_sq.sqrt().add_(self.safe_guard)
                    group['grad_variant'][p_name] = exp_avg_first_moment_grad / denom

    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr
    
    @abstractclassmethod
    def step(self, loss=None):
        raise NotImplementedError