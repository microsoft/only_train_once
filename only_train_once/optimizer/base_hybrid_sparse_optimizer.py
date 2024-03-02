from abc import ABC, abstractclassmethod
from torch.optim.optimizer import Optimizer, required
import torch
from .base_optimizer import BaseOptimizer
from .hyperparameter import DEFAULT_OPT_PARAMS, SUPPORT_GRADIENT_ESTIMATES
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation, index_transformation_param_group
from .importance_score import calculate_importance_score
import numpy as np

class SparseOptimizerMetrics:
    num_groups = 0
    num_zero_groups = 0
    num_important_groups = 0
    num_redundant_groups = 0
    
    # For CRIC
    num_violating_groups = 0
    num_trial_violating_groups = 0
    num_historical_violating_groups = 0
    
    norm_violating_groups = 0.0

    norm_params = 0.0
    norm_important_groups = 0.0
    norm_redundant_groups = 0.0

    group_sparsity = 0.0

class BaseHybridSparseOptimizer(BaseOptimizer):
    def __init__(self, params, variant='sgd', lr=required, \
                 first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.0, group_divisible=1, additional_defaults=dict()):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if variant not in SUPPORT_GRADIENT_ESTIMATES:
            raise ValueError("Need to select a gradient estimation from {}".format(SUPPORT_GRADIENT_ESTIMATES))
        
        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']

        defaults = dict(lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, grad_variant=dict(), global_start_idx=0, global_idx=0)
        defaults.update(additional_defaults)

        super(BaseHybridSparseOptimizer, self).__init__(params, defaults)

        # Set up total number of prunable groups
        self.total_num_groups = 0
        for param_group in params:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                if param_group['num_groups'] <= group_divisible:
                    param_group['is_prunable'] = False
                else:
                    self.total_num_groups += param_group['num_groups']

        self.group_divisible = group_divisible
        self.target_group_sparsity = target_group_sparsity
        self.target_num_redundant_groups = int(self.total_num_groups * min(self.target_group_sparsity, 0.999))
        self.opt_metrics = SparseOptimizerMetrics()

        self.auxiliary_param_groups = dict()
        for group in self.param_groups:
            if group['is_auxiliary']:
                self.auxiliary_param_groups[group['id']] = group
        
    def gradient_descent_step(self, param_group):
        for p_name, p in zip(param_group['p_names'], param_group['params']):
            if p_name not in param_group['grad_variant']:
                continue
            if param_group['weight_decay'] is not None and param_group['variant'] == 'adamw':
                p.data.add_(param_group['weight_decay'] * p.data, alpha=-param_group['lr'])
            p.data.add_(param_group['grad_variant'][p_name], alpha=-param_group['lr'])

    def fix_pruned_groups_as_zeros(self, param_group):
        if len(param_group['pruned_idxes']) > 0:
            for p, p_transform in zip(param_group['params'], param_group['p_transform']):
                pruned_idxes = index_transformation_param_group(param_group['pruned_idxes'], p_transform, param_group)
                if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                    p.data[:, pruned_idxes] = 0.0
                else:
                    p.data[pruned_idxes] = 0.0
                    
            # Tackle auxiliary params
            for ng_id, offset in param_group['auxiliary_ngs']:
                pruned_aux_idxes = [i + offset for i in pruned_idxes]
                for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                    if aux_p.grad is None:
                        continue
                    aux_p.data[pruned_aux_idxes, ...] = 0.0

    def compute_importance_scores(self, **kwargs):
        global_start_idx = 0
        self.global_scores = list() # Accumulate global scores
        # Calculate raw importance scores by varying criteria
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                calculate_importance_score(self.importance_score_criteria, group)

        # Normalize importance_score
        # Calculate normalization_denoms
        normalization_denoms = dict.fromkeys(self.importance_score_criteria.keys(), self.safe_guard)
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                for proxy_name in self.importance_score_criteria:
                    if not proxy_name in group['importance_scores']:
                        continue
                    normalization_denoms[proxy_name] += torch.sum(group['importance_scores'][proxy_name] ** 2, dim=0).item()
        for proxy_name in normalization_denoms:
            normalization_denoms[proxy_name] = np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard

        global_start_idx = 0
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['importance_scores']['overall'] = None
                for proxy_name in self.importance_score_criteria:
                    if not proxy_name in group['importance_scores']:
                        continue
                    group['importance_scores'][proxy_name].mul_(self.importance_score_criteria[proxy_name] / normalization_denoms[proxy_name])
                    if group['importance_scores']['overall'] is None:
                        group['importance_scores']['overall'] = group['importance_scores'][proxy_name].clone()
                    else:
                        group['importance_scores']['overall'] += group['importance_scores'][proxy_name]
                group['global_start_idx'] = global_start_idx
                group['global_idxes'] = np.arange(global_start_idx, global_start_idx+group['num_groups'])
                global_start_idx += group['num_groups']
                self.global_scores.append(group['importance_scores']['overall'])

    def compute_metrics(self):
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0

        for group in self.param_groups:
            if not (group['is_prunable'] and not group['is_auxiliary']):
                continue
            norm_group = None
            import_idxes = group['important_idxes']
            redund_idxes = group['active_redundant_idxes'] + group['pruned_idxes']

            for param, p_transform in zip(group['params'], group['p_transform']):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'], group['num_heads'])
                else:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'])
                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            norm_group = torch.sqrt(norm_group)
            self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()
            self.opt_metrics.norm_params += torch.sum(norm_group).item()
            self.opt_metrics.norm_important_groups += torch.sum(norm_group[import_idxes]).item()
            self.opt_metrics.norm_redundant_groups += torch.sum(norm_group[redund_idxes]).item()
            self.opt_metrics.num_important_groups += len(import_idxes)
            self.opt_metrics.num_redundant_groups += len(redund_idxes)

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(self.total_num_groups + self.safe_guard)

        return self.opt_metrics