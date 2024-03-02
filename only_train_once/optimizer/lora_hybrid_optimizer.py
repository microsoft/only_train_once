import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

from .hyperparameter import DEFAULT_OPT_PARAMS
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

class LORAHYBRIDOPT(Optimizer):
    def __init__(self, params, variant='sgd', lr=required, warm_up_steps=100, training_steps=100, redundant_group_idxes=dict(), \
                 first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, fix_zero_groups=True):

        print("Setup HYBRIDOPT")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.num_steps = 0
        self.warm_up_steps = warm_up_steps
        
        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']
        
        self.norm_important_groups = 0.0 # norm for important groups
        self.norm_redundant_groups = 0.0 # norm for redundant groups
        self.num_important_groups = 0 # number of important groups
        self.num_redundant_groups = 0 # number of redundant groups
        
        self.redundant_group_idxes = redundant_group_idxes
        self.training_steps = training_steps

        defaults = dict(lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, grad_variant=dict())
        
        super(LORAHYBRIDOPT, self).__init__(params, defaults)
        
        self.first_moment_grads = dict()
        self.second_moment_grads = dict()

        # Set up total number of prunable groups
        self.total_num_groups = 0
        for param_group in self.param_groups:
            if param_group['is_prunable']:
                if any(['embed' in p_name for p_name in param_group['p_names']]):
                    print(param_group['p_names'])
                    param_group['num_groups'] = 4096
                self.total_num_groups += param_group['num_groups']

        self.set_redundant_bool(redundant_group_idxes)

        self.kt_steps = self.training_steps - self.warm_up_steps

        self.safe_guard = 1e-8

        # Create param dictionary for facilitating accessing lora_A modules
        self.named_parameters = dict()
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.named_parameters[p_name] = param

    def set_redundant_bool(self, redundant_group_idxes=dict()):
        num_active_redundant_idxes = 0
        for group in self.param_groups:
            if not group['is_prunable']:
                continue
            
            group['active_redundant_idxes'] = redundant_group_idxes[group['id']]
            group['active_redundant_bool'] = torch.zeros(group['num_groups'], dtype=torch.bool).cuda()
            group['active_redundant_bool'][group['active_redundant_idxes']] = True
        
    def __setstate__(self, state):
        super(LORAHYBRIDOPT, self).__setstate__(state)
                    
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

    def terminate(self):
        if self.num_steps > self.training_steps:
            return True
        else:
            return False

    def step(self):
        self.num_steps += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # Second pass to update variables
        t = (self.num_steps - self.warm_up_steps) % self.kt_steps
        decay_ratio = 1.0
        if self.num_steps > self.warm_up_steps:
            if (self.num_steps - self.warm_up_steps) // self.kt_steps >= 1:
                decay_ratio = 0.0
            else:
                decay_ratio = max(0, (self.kt_steps - t - 1.0) / (self.kt_steps - t))
        else:
            decay_ratio = 1.0
        
        for i, group in enumerate(self.param_groups):
            if not group['is_prunable']:
                continue
            if self.num_steps <= self.warm_up_steps:
                for p_name, p in zip(group['p_names'], group['params']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
            else:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if 'lora_B' in p_name:
                        if group['weight_decay'] is not None and group['variant'] == 'adamw':
                            p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                        p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                        original_weight_name = p_name.split('lora_B')[0] + 'weight'
                        original_bias_name = p_name.split('lora_B')[0] + 'bias'
                        original_weight = self.named_parameters[original_weight_name]
                        original_bias = None if original_bias_name not in self.named_parameters else self.named_parameters[original_bias_name]
                        active_redundant_bool = None
                        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                            active_redundant_bool = tensor_transformation(group['active_redundant_bool'], TensorTransform.REVERSE_MULTIHEAD_HEADDIM, \
                                                                          num_groups=group['num_groups'], num_heads=group['num_heads'])
                        elif p_transform == TensorTransform.MULTIHEAD_NUMHEAD:
                            active_redundant_bool = tensor_transformation(group['active_redundant_bool'], TensorTransform.REVERSE_MULTIHEAD_NUMHEAD, \
                                                                          num_groups=group['num_groups'], head_dim=group['head_dim'])
                        else:
                            active_redundant_bool = group['active_redundant_bool']
                        p.data[active_redundant_bool] *= decay_ratio
                        original_weight.data[active_redundant_bool] *= decay_ratio
                        if original_bias is not None:
                            original_bias.data[active_redundant_bool] *= decay_ratio
                    if 'lora_embedding_B' in p_name:
                        if group['weight_decay'] is not None and group['variant'] == 'adamw':
                            p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                        p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                        for (decay_p_name, decay_param, decay_p_transform) in zip(group['p_names'], group['params'], group['p_transform']):   
                            if decay_p_transform == TensorTransform.BASIC:
                                decay_param.data[group['active_redundant_bool']] *= decay_ratio
                            elif decay_p_transform == TensorTransform.TRANSPOSE:
                                decay_param.data[:, group['active_redundant_bool']] *= decay_ratio
                        break
        return 

            # if 'num_heads' in group and group['num_heads'] > 1 and not any(['embed' in p_name for p_name in group['p_names']]):
            #     group['active_redundant_bool'] = tensor_transformation(group['active_redundant_bool'], TensorTransform.REVERSE_MULTIHEAD, \
            #                                                             num_groups=group['num_groups'], num_heads=group['num_heads']) 
            # num_active_redundant_idxes += len(group['active_redundant_idxes'])
        
    def compute_norm_groups(self):
        self.norm_important_groups = 0.0
        self.norm_redundant_groups = 0.0
        self.num_important_groups = 0
        self.num_redundant_groups = 0
        
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                id = group['id']
                norm_group = None
                for p_name, param, p_transform in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_transform == TensorTransform.NO_PRUNE:
                        continue
                    param_transform = None
                    if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                        param_transform = tensor_transformation(param, p_transform, group['num_groups'], group['num_heads'])
                    else:
                        param_transform = tensor_transformation(param, p_transform, group['num_groups'])
                    if norm_group == None:
                        norm_group = torch.norm(param_transform, dim=1) ** 2
                    else:
                        norm_group += torch.norm(param_transform, dim=1) ** 2
                norm_group = torch.sqrt(norm_group)
                redund_idxes = group['active_redundant_idxes']
                import_idxes = [i for i in range(norm_group.shape[0]) if i not in redund_idxes]
                self.norm_important_groups += torch.sum(norm_group[import_idxes]).item()
                self.norm_redundant_groups += torch.sum(norm_group[redund_idxes]).item()
                self.num_important_groups += len(import_idxes)
                self.num_redundant_groups += len(redund_idxes)

        return self.norm_important_groups, self.norm_redundant_groups, self.num_important_groups, self.num_redundant_groups 

    def compute_group_sparsity_param_norm(self):
        total_num_zero_groups = 0
        norm_x = 0.0
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                norm_group = None
                for p_name, param, p_transform in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_transform == TensorTransform.NO_PRUNE:
                        continue
                    param_transform = None
                    if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                        param_transform = tensor_transformation(param, p_transform, group['num_groups'], group['num_heads'])
                    else:
                        param_transform = tensor_transformation(param, p_transform, group['num_groups'])
                    if norm_group == None:
                        norm_group = torch.norm(param_transform, dim=1) ** 2
                    else:
                        norm_group += torch.norm(param_transform, dim=1) ** 2
                norm_group = torch.sqrt(norm_group)
                num_zero_groups = torch.sum(norm_group == 0).item()
                total_num_zero_groups += num_zero_groups
                norm_x += torch.sum(norm_group).item()
        group_sparsity = total_num_zero_groups / float(self.total_num_groups + self.safe_guard)
        return group_sparsity, norm_x, total_num_zero_groups

    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr