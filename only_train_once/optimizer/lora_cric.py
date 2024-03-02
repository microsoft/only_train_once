import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from collections import defaultdict

from .hyperparameter import DEFAULT_OPT_PARAMS
from .importance_score import calculate_importance_score_lora
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

class LORACRIC(Optimizer):
    def __init__(self, params, variant='sgd', lr=required, \
                 first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.5, tolerance=0, \
                 start_sampling_step=0, sampling_steps=None, sampling_periods=None, \
                 lora_update_freq=4, importance_score_criteria=None):

        print("Setup CRIC")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.num_steps = 0
        self.start_sampling_step = start_sampling_step
        self.sampling_steps = sampling_steps
        self.sampling_periods = int(max(1, sampling_periods)) # How many periods that the pruning last for.
        self.curr_sampling_period = -1 # Track pruning periodp
        
        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']
        
        self.redundant_groups_identified = False
        self.lora_update_freq = lora_update_freq
        self.tolerance = tolerance
        
        self.safe_guard = 1e-8
        self.target_group_sparsity = target_group_sparsity
        
        self.norm_important_groups = 0.0 # norm for important groups
        self.norm_redundant_groups = 0.0 # norm for redundant groups
        self.num_important_groups = 0 # number of important groups
        self.num_redundant_groups = 0 # number of redundant groups
        
        self.redundant_group_idxes = dict()

        self.importance_score_criteria = importance_score_criteria

        defaults = dict(lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, grad_variant=dict(), global_start_idx=0, global_idx=0, \
                        active_violating_idxes=list(), trial_violating_idxes=list(), historical_violating_idxes=list())
        
        super(LORACRIC, self).__init__(params, defaults)
        
        self.first_moment_grads = dict()
        self.second_moment_grads = dict()
        
        # Set up total number of prunable groups
        self.total_num_groups = 0
        self.total_num_groups_by_clusters = dict()
        self.prunable_param_group_clusters = dict()

        self.target_group_sparsity = self.target_group_sparsity if isinstance(self.target_group_sparsity, dict) else {'overall': self.target_group_sparsity}

        if isinstance(self.target_group_sparsity, dict):
            for cluster_name in self.target_group_sparsity:
                self.prunable_param_group_clusters[cluster_name] = list()
                for param_group in self.param_groups:
                    if not param_group['is_prunable']:
                        continue
                    in_cluster = False
                    for p_name in param_group['p_names']:
                        if cluster_name in p_name:
                            in_cluster = True
                            break
                    if in_cluster:
                        self.prunable_param_group_clusters[cluster_name].append(param_group)
        
        for cluster_name in self.prunable_param_group_clusters:
            param_group_cluster = self.prunable_param_group_clusters[cluster_name]
            self.total_num_groups_by_clusters[cluster_name] = 0
            for param_group in param_group_cluster:
                self.total_num_groups += param_group['num_groups']
                self.total_num_groups_by_clusters[cluster_name] += param_group['num_groups']

        print(self.total_num_groups, self.total_num_groups_by_clusters)

        # Set up target number of redundant groups
        self.target_num_redundant_groups = 0
        self.target_num_redundant_groups_by_clusters = dict()

        for cluster_name in self.prunable_param_group_clusters:
            param_group_cluster = self.prunable_param_group_clusters[cluster_name]
            self.target_num_redundant_groups_by_clusters[cluster_name] = 0
            for param_group in param_group_cluster:
                self.target_num_redundant_groups_by_clusters[cluster_name] = int(self.total_num_groups_by_clusters[cluster_name] \
                                                                              * min(self.target_group_sparsity[cluster_name], 0.999))
            self.target_num_redundant_groups += self.target_num_redundant_groups_by_clusters[cluster_name]
        print(self.target_num_redundant_groups, self.target_num_redundant_groups_by_clusters)

        # Create param dictionary for facilitating accessing lora_A modules
        self.named_parameters = dict()
        self.cache_parameters = dict()
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.named_parameters[p_name] = param
                self.cache_parameters[p_name] = param.data.clone()        
            if param_group['is_prunable']:
                param_group['importance_score_collection'] = dict()
                param_group['active_violating_idxes_collection'] = dict()
                param_group['loss_collection'] = dict()
                for sample_period in range(self.sampling_periods + 1):
                    param_group['importance_score_collection'][sample_period] = list()
                    param_group['active_violating_idxes_collection'][sample_period] = list()
                    param_group['loss_collection'][sample_period] = list()

    def __setstate__(self, state):
        super(LORACRIC, self).__setstate__(state)
    
    def reset_params(self):
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                if 'lora' not in p_name:
                    self.named_parameters[p_name].data.copy_(self.cache_parameters[p_name])
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()
                    
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

    def compute_num_active_violating_groups(self):
        num_violating_groups = 0
        for param_group in self.param_groups:
            num_violating_groups += len(param_group['active_violating_idxes'])
        return num_violating_groups

    def terminate(self):
        if self.curr_sampling_period >= self.sampling_periods:
            return True
        if self.curr_sampling_period >= 0 and self.compute_num_active_violating_groups() <= self.tolerance:
            return True

    def compute_importance_scores(self, sample_period=0, loss=None):
        global_start_idx = 0
        self.global_scores = list() # Accumulate global scores
        # Calculate raw importance scores by varying criteria
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                calculate_importance_score_lhspg(self.importance_score_criteria, group, self.named_parameters)

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

        self.cluster_importance_scores = dict()
        for cluster_name in self.prunable_param_group_clusters:
            param_group_cluster = self.prunable_param_group_clusters[cluster_name]
            global_start_idx = 0
            cluster_importance_score = list()
            for group in param_group_cluster:
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
                    cluster_importance_score.append(group['importance_scores']['overall'])
                    group['importance_score_collection'][sample_period].append(group['importance_scores'])

                    if len(group['active_violating_idxes']) > 0:
                        group['loss_collection'][sample_period].append(loss)
                        group['active_violating_idxes_collection'][sample_period] = [i for i in group['active_violating_idxes']]
                        
            self.cluster_importance_scores[cluster_name] = cluster_importance_score

    def update_violating_set(self, sample_period=0):
        print("update_violating_set")
        for cluster_name in self.prunable_param_group_clusters:
            if len(self.cluster_importance_scores[cluster_name]) == 0:
                continue
            cluster_importance_score = torch.cat(self.cluster_importance_scores[cluster_name], dim=0)
            curr_K = self.target_num_redundant_groups_by_clusters[cluster_name]

            _, top_indices = torch.topk(-cluster_importance_score, curr_K)
            top_indices = top_indices.cpu().numpy()
            top_indices = top_indices.tolist()

            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    if sample_period == 0:
                        # line 4 in CRIC
                        group['active_violating_idxes'] = (global_trial_violating_idxes - group['global_start_idx']).tolist()
                    else:
                        # line 16 in CRIC
                        group['active_violating_idxes'] = [
                            i for i in group['trial_violating_idxes'] if i not in group['historical_violating_idxes']
                        ]
                    group['active_violating_bool'] = torch.zeros(group['num_groups'], dtype=torch.bool).cuda()
                    group['active_violating_bool'][group['active_violating_idxes']] = True

    def update_trial_violating_set(self):
        for cluster_name in self.prunable_param_group_clusters:
            if len(self.cluster_importance_scores[cluster_name]) == 0:
                continue
            cluster_importance_score = torch.cat(self.cluster_importance_scores[cluster_name], dim=0)
            curr_K = self.target_num_redundant_groups_by_clusters[cluster_name]
            
            _, top_indices = torch.topk(-cluster_importance_score, curr_K)
            top_indices = top_indices.cpu().numpy()
            top_indices = top_indices.tolist()

            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    # line 14 in CRIC
                    group['trial_violating_idxes'] += (global_trial_violating_idxes - group['global_start_idx']).tolist()
                    group['trial_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['active_violating_idxes']]
    
    def update_historical_violating_set(self):
        # line 8 in CRIC
        for param_group in self.param_groups:
            param_group['historical_violating_idxes'] += param_group['active_violating_idxes']

    def step(self, loss=None):
        self.num_steps += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()
        
        if self.num_steps > self.start_sampling_step and self.curr_sampling_period < self.sampling_periods: 
            if (self.num_steps - self.start_sampling_step - 1) % self.sampling_steps == 0:
                self.curr_sampling_period += 1
                print("Current sampling period", self.curr_sampling_period)
                self.compute_importance_scores(self.curr_sampling_period, loss)  
                self.update_violating_set(self.curr_sampling_period)
                self.update_historical_violating_set()
                self.reset_params()
            else:
                self.compute_importance_scores(self.curr_sampling_period, loss)
            self.update_trial_violating_set()

        # Second pass to update variables
        t = (self.num_steps - self.start_sampling_step - 1) % self.sampling_steps
        for i, group in enumerate(self.param_groups):
            if not group['is_prunable']:
                continue
            if len(group['active_violating_idxes']) == 0:
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
                        active_violating_bool = None
                        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                            active_violating_bool = tensor_transformation(group['active_violating_bool'], TensorTransform.REVERSE_MULTIHEAD_HEADDIM, \
                                                                          num_groups=group['num_groups'], num_heads=group['num_heads'])
                        elif p_transform == TensorTransform.MULTIHEAD_NUMHEAD:
                            active_violating_bool = tensor_transformation(group['active_violating_bool'], TensorTransform.REVERSE_MULTIHEAD_NUMHEAD, \
                                                                          num_groups=group['num_groups'], head_dim=group['head_dim'])
                        else:
                            active_violating_bool = group['active_violating_bool']
                        p.data[active_violating_bool] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                        original_weight.data[active_violating_bool] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                        if original_bias is not None:
                            original_bias.data[active_violating_bool] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)

                    if 'lora_embedding_B' in p_name:
                        if group['weight_decay'] is not None and group['variant'] == 'adamw':
                            p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                        p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                        for (decay_p_name, decay_param, decay_p_transform) in zip(group['p_names'], group['params'], group['p_transform']):   
                            if decay_p_transform == TensorTransform.BASIC:
                                decay_param.data[group['active_violating_bool']] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                            elif decay_p_transform == TensorTransform.TRANSPOSE:
                                decay_param.data[:, group['active_violating_bool']] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                        break
        return 
        
    def compute_norm_groups(self):
        self.norm_important_groups = 0.0
        self.norm_violating_groups = 0.0
        self.num_important_groups = 0
        self.num_violating_groups = 0
        
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
                if norm_group is None:
                    continue
                norm_group = torch.sqrt(norm_group)
                violating_idxes = group['active_violating_idxes']
                import_idxes = [i for i in range(norm_group.shape[0]) if i not in violating_idxes]
                self.norm_important_groups += torch.sum(norm_group[import_idxes]).item()
                self.norm_violating_groups += torch.sum(norm_group[violating_idxes]).item()
                self.num_important_groups += len(import_idxes)
                self.num_violating_groups += len(violating_idxes)
        return self.norm_important_groups, self.norm_violating_groups, self.num_important_groups, self.num_violating_groups 
    
    def compute_accumulate_saliency_score(self):
        active_violating_sets = defaultdict(int)
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                for sample_period in param_group['active_violating_idxes_collection']:
                    active_violating_sets[sample_period] += len(param_group['active_violating_idxes_collection'][sample_period])

        # Compute accumulated saliency score
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                param_group['accumulated_importance_score'] = None
                accumulate_count = 0
                for sample_period in param_group['importance_score_collection']:
                    importance_score_sample_steps = param_group['importance_score_collection'][sample_period]
                    for sample_step, importance_score in enumerate(importance_score_sample_steps):
                        if param_group['accumulated_importance_score'] is None:
                            param_group['accumulated_importance_score'] = importance_score['overall'].clone()
                        else:
                            param_group['accumulated_importance_score'] += importance_score['overall']
                        accumulate_count += 1
                if param_group['accumulated_importance_score'] is not None:
                    param_group['accumulated_importance_score'] /= float(accumulate_count)
                    
                for sample_period in param_group['loss_collection']:
                    if len(param_group['loss_collection'][sample_period]) > 0:
                        losses = param_group['loss_collection'][sample_period]
                        # print(sample_period, losses)

                # print("violating idxes")
                for sample_period in param_group['active_violating_idxes_collection']:
                    if len(param_group['active_violating_idxes_collection'][sample_period]) > 0:
                        violating_idxes = param_group['active_violating_idxes_collection'][sample_period]
                        loss_scores = param_group['loss_collection'][sample_period]
                        avg_loss_score = sum(loss_scores) / len(loss_scores) / float(active_violating_sets[sample_period])
                        param_group['accumulated_importance_score'][violating_idxes] += self.importance_score_criteria['loss'] * avg_loss_score

    def identify_redundant_groups(self):
        cluster_importance_scores = dict()
        for cluster_name in self.prunable_param_group_clusters:
            param_group_cluster = self.prunable_param_group_clusters[cluster_name]
            global_start_idx = 0
            cluster_importance_score = list()
            for group in param_group_cluster:
                if group['is_prunable'] and not group['is_auxiliary']:
                    group['global_start_idx'] = global_start_idx
                    group['global_idxes'] = np.arange(global_start_idx, global_start_idx+group['num_groups'])
                    global_start_idx += group['num_groups']
                    cluster_importance_score.append(group['accumulated_importance_score'])
            cluster_importance_scores[cluster_name] = cluster_importance_score
        
        for cluster_name in cluster_importance_scores:
            if len(cluster_importance_scores[cluster_name]) == 0:
                continue
            cluster_importance_score = torch.cat(self.cluster_importance_scores[cluster_name], dim=0)
            target_num_redundant_groups = self.target_num_redundant_groups_by_clusters[cluster_name]
            
            _, top_indices = torch.topk(-cluster_importance_score, target_num_redundant_groups)
            top_indices = top_indices.cpu().numpy()
            
            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])
                    self.redundant_group_idxes[group['id']] = (global_active_redundant_idx - group['global_start_idx']).tolist()
        
        return self.redundant_group_idxes
    
    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr