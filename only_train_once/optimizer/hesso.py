import torch
import numpy as np
from torch.optim.optimizer import required

from .importance_score import calculate_importance_score
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation, index_transformation_param_group
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer

class HESSO(BaseHybridSparseOptimizer):
    '''
    HESSO: Hybrid Efficient Structured Sparse Optimizer
    '''
    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, \
                 dampening=None, weight_decay=None, target_group_sparsity=0.5, \
                 start_pruning_step=0, pruning_steps=None, pruning_periods=1, \
                 group_divisible=1, importance_score_criteria='default', device='cuda'):

        print("Setup HESSO")
        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(max(1, pruning_periods)) # How many periods that the pruning last for.
        self.pruning_steps = pruning_steps
        self.pruning_period_duration = self.pruning_steps // self.pruning_periods # How many pruning steps for each period
        self.curr_pruning_period = 0 # Track pruning period
        self.device = device
        
        self.pruned_group_idxes = list()
        
        if importance_score_criteria == 'default':
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2,\
                                              'cosine_similarity': 0.2, \
                                              'taylor_first_order': 0.2, 'taylor_second_order': 0.2}
        else:
            self.importance_score_criteria = importance_score_criteria

        super(HESSO, self).__init__(params=params, variant=variant, lr=lr, first_momentum=first_momentum, second_momentum=second_momentum, \
                                    dampening=dampening, weight_decay=weight_decay, target_group_sparsity=target_group_sparsity, \
                                    group_divisible=group_divisible)

        for param_group in self.param_groups:
            param_group['important_idxes'] = [i for i in range(param_group['num_groups'])]
            param_group['active_redundant_idxes'] = list()
            param_group['pruned_idxes'] = list()
            param_group['importance_scores'] = dict()
            
        self.active_num_redundant_groups = list()
        # Set up active number redundant groups for each pruning period
        groups_sum = 0
        for p in range(self.pruning_periods):
            if p == self.pruning_periods - 1:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                groups_sum += self.active_num_redundant_groups[p]
        print("Target redundant groups per period: ", self.active_num_redundant_groups)

    def identify_redundant_groups(self):
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_active_num_redundant_groups = self.active_num_redundant_groups[self.curr_pruning_period]
        curr_K = len(self.pruned_group_idxes) + curr_active_num_redundant_groups
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy()
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[:curr_active_num_redundant_groups].tolist()
        self.pruned_group_idxes.extend(top_indices)

        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])
                group['active_redundant_idxes'] = (global_active_redundant_idx - group['global_start_idx']).tolist()
                # Refine important_idx by group_divisible
                if group['num_groups'] < self.group_divisible:
                    group['active_redundant_idxes'].clear()
                    group['pruned_idxes'].clear()
                else:
                    curr_num_important_groups = len(group['important_idxes'])
                    trial_num_important_groups = curr_num_important_groups - len(group['active_redundant_idxes'])                    
                    if trial_num_important_groups % self.group_divisible != 0 or trial_num_important_groups <= 0:
                        ratio = trial_num_important_groups // self.group_divisible + 1 # Add one will preserve more groups, otherwise will slim more.
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(int(self.group_divisible), 1)
                        else:
                            refined_num_important_groups = max(int(ratio * self.group_divisible), int(self.group_divisible))
                        refined_num_important_groups = min(group['num_groups'], refined_num_important_groups)
                        refined_num_active_redundant_groups = group['num_groups'] - len(group['pruned_idxes']) - refined_num_important_groups
                        self.target_num_redundant_groups += (refined_num_active_redundant_groups - len(group['active_redundant_idxes']))
                        group['active_redundant_idxes'] = group['active_redundant_idxes'][:refined_num_active_redundant_groups]
                group['important_idxes'] = [i for i in group['important_idxes'] if (i not in group['active_redundant_idxes'] and i not in group['pruned_idxes'])]
 
    def commit_redundant_idxes(self):
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['pruned_idxes'].extend(group['active_redundant_idxes'].copy())
                group['active_redundant_idxes'].clear()
                group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['pruned_idxes']]
                group['importance_scores'].clear()

    def step(self, loss=None, closure=None):
        if closure is not None:
            loss = closure()

        self.num_steps += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # Partition groups into important and redundant groups  
        if self.num_steps >= self.start_pruning_step and self.curr_pruning_period < self.pruning_periods:
            if (self.num_steps - self.start_pruning_step - 1) % self.pruning_period_duration == 0:
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1

        # Second pass to update variables
        t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        for group in self.param_groups:
            if not group['is_prunable'] or len(group['active_redundant_idxes']) == 0:
                self.gradient_descent_step(group)
            elif group['is_prunable'] and len(group['active_redundant_idxes']) > 0:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])

                    active_redundant_idxes = index_transformation_param_group(group['active_redundant_idxes'], p_transform, group)
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, active_redundant_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
                    else:
                        p.data[active_redundant_idxes] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)

                    # Tackle auxiliary params
                    for ng_id, offset in group['auxiliary_ngs']:
                        active_redundant_aux_idxes = [i + offset for i in active_redundant_idxes]
                        for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[active_redundant_aux_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)

            self.fix_pruned_groups_as_zeros(group)

        if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
            self.commit_redundant_idxes()