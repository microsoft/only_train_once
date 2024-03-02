import torch
import numpy as np
# from torch.optim.optimizer import Optimizer, required
from torch.optim.optimizer import required
from .base_optimizer import BaseOptimizer
from .hyperparameter import DEFAULT_OPT_PARAMS

from .importance_score import calculate_importance_score_dhspg
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation

class HESSOCRIC(BaseOptimizer):
    def __init__(self, params, variant='sgd', lr=required, \
                 first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.5, tolerance=0, \
                 start_sampling_step=0, sampling_steps=None, sampling_periods=None, hybrid_steps=None, \
                 importance_score_criteria='default'):

        print("Setup HESSOCRIC")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.num_steps = 0
        self.start_sampling_step = start_sampling_step
        self.sampling_steps = sampling_steps
        self.sampling_periods = int(max(1, sampling_periods)) # How many periods that the pruning last for.
        self.curr_sampling_period = -1 # Track pruning periodp
        self.hybrid_steps = hybrid_steps
        
        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']
        
        self.redundant_groups_identified = False
        self.tolerance = tolerance
        
        self.safe_guard = 1e-8
        self.target_group_sparsity = target_group_sparsity

        if importance_score_criteria == 'default' or importance_score_criteria is None:
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2,\
                                              'cosine_similarity': 0.2, \
                                              'taylor_first_order': 0.2, 'taylor_second_order': 0.2, 'loss': 1.0}
        else:
            self.importance_score_criteria = importance_score_criteria

        defaults = dict(lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, grad_variant=dict(), global_start_idx=0, global_idx=0, \
                        active_violating_idxes=list(), trial_violating_idxes=list(), historical_violating_idxes=list(), redundant_idxes=list())

        super(HESSOCRIC, self).__init__(params, defaults)

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
                    if in_cluster or cluster_name == 'overall':
                        self.prunable_param_group_clusters[cluster_name].append(param_group)

        for cluster_name in self.prunable_param_group_clusters:
            param_group_cluster = self.prunable_param_group_clusters[cluster_name]
            self.total_num_groups_by_clusters[cluster_name] = 0
            for param_group in param_group_cluster:
                self.total_num_groups += param_group['num_groups']
                self.total_num_groups_by_clusters[cluster_name] += param_group['num_groups']

        print("Total number of groups")
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
        print("Target number of redundant groups")
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

        self.prunable_param_group_dict = dict()
        self.num_prunable_param_groups = 0
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                self.num_prunable_param_groups += 1
                self.prunable_param_group_dict[param_group['id']] = param_group
        self.param_group_ids = list(self.prunable_param_group_dict.keys())
        self.trial_group_sparsties = [0.25, 0.5, 0.75]
        self.start_global_sampling_step = 2 * len(self.trial_group_sparsties) * self.num_prunable_param_groups + self.start_sampling_step

        self.is_cric_terminated = False

    def reset_cache_params(self):
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.cache_parameters[p_name] = param.data.clone()  

    def reset_params(self):
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.named_parameters[p_name].data.copy_(self.cache_parameters[p_name])
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()

    def compute_num_active_violating_groups(self):
        num_violating_groups = 0
        for param_group in self.param_groups:
            num_violating_groups += len(param_group['active_violating_idxes'])
        return num_violating_groups

    def cric_terminate(self):
        if self.curr_sampling_period >= self.sampling_periods:
            return True
        elif self.curr_sampling_period >= 1 and self.compute_num_active_violating_groups() <= self.tolerance:
            return True
        else:
            return False

    def compute_importance_scores(self, sample_period=0, loss=None):
        global_start_idx = 0
        self.global_scores = list() # Accumulate global scores
        # Calculate raw importance scores by varying criteria
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                calculate_importance_score_dhspg(self.importance_score_criteria, group)

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
                        # group['loss_collection'][sample_period].append(loss.item() / self.ref_loss)
                        group['active_violating_idxes_collection'][sample_period] = [i for i in group['active_violating_idxes']]

            self.cluster_importance_scores[cluster_name] = cluster_importance_score

    def update_violating_set(self, sample_period=0):
        print("Update violating set")
        for cluster_name in self.prunable_param_group_clusters:
            if len(self.cluster_importance_scores[cluster_name]) == 0:
                continue
            cluster_importance_score = torch.cat(self.cluster_importance_scores[cluster_name], dim=0)
            curr_K = self.target_num_redundant_groups_by_clusters[cluster_name]

            _, top_indices = torch.topk(-cluster_importance_score, curr_K)
            top_indices = top_indices.cpu().numpy().tolist()

            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    if sample_period == 1:
                        # line 4 in CRIC
                        group['active_violating_idxes'] = (global_trial_violating_idxes - group['global_start_idx']).tolist()
                    else:
                        # line 16 in CRIC
                        group['active_violating_idxes'] = [
                            i for i in group['trial_violating_idxes'] if i not in group['historical_violating_idxes']
                        ]

    def update_trial_violating_set(self):
        for cluster_name in self.prunable_param_group_clusters:
            if len(self.cluster_importance_scores[cluster_name]) == 0:
                continue
            cluster_importance_score = torch.cat(self.cluster_importance_scores[cluster_name], dim=0)
            curr_K = self.target_num_redundant_groups_by_clusters[cluster_name]

            _, top_indices = torch.topk(-cluster_importance_score, curr_K)
            top_indices = top_indices.cpu().numpy().tolist()

            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    # line 14 in CRIC
                    group['trial_violating_idxes'] += (global_trial_violating_idxes - group['global_start_idx']).tolist()
                    group['trial_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['active_violating_idxes']]
                    group['trial_violating_idxes'] = list(set(group['trial_violating_idxes']))

    def update_historical_violating_set(self):
        # line 8 in CRIC
        for param_group in self.param_groups:
            param_group['historical_violating_idxes'] += param_group['active_violating_idxes']
            param_group['historical_violating_idxes'] = list(set(param_group['historical_violating_idxes']))

    def proj_trial_group_sparsity(self, param_group, trial_group_sparsity):
        if 'importance_scores' not in param_group:
            return
        
        num_redund_grps = max(min(int(param_group['num_groups'] * trial_group_sparsity), param_group['num_groups']), 1)
        _, top_indices = torch.topk(-param_group['importance_scores']['overall'], num_redund_grps)
        top_indices = top_indices.cpu().numpy().tolist()
        for (p, p_transform) in zip(param_group['params'], param_group['p_transform']):
            if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                p.data[:, top_indices, ...] = 0.0
            else:
                p.data[top_indices] = 0.0

    def cric_step(self):
        # print("Current sampling period", self.curr_sampling_period, self.num_steps)
        self.compute_importance_scores(self.curr_sampling_period)
        if (self.num_steps - self.start_global_sampling_step) % self.sampling_steps == 0:
            self.curr_sampling_period += 1
            self.update_violating_set(self.curr_sampling_period)
            self.update_historical_violating_set()
            self.reset_params()
        self.update_trial_violating_set()
        
        # Second pass to update variables
        t = (self.num_steps - self.start_global_sampling_step - 1) % self.sampling_steps
        for group in self.param_groups:
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
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, group['active_violating_idxes'], ...] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                    else:
                        p.data[group['active_violating_idxes']] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)

                    # Tackle auxiliary params
                    for ng_id, offset in group['auxiliary_ngs']:
                        aux_pg = self.auxiliary_param_groups[ng_id]
                        for aux_p in aux_pg['params']:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[offset:offset+group['num_groups'], ...] *= (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                            
    def basic_step(self):
        for group in self.param_groups:
            for p_name, p in zip(group['p_names'], group['params']):
                if p_name not in group['grad_variant']:
                    continue
                if group['weight_decay'] is not None and group['variant'] == 'adamw':
                    p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
    
    def proj_step(self, loss):
        self.compute_importance_scores(self.curr_sampling_period, loss)
        curr_param_group_idx = (self.num_steps - self.start_sampling_step) // (2 * len(self.trial_group_sparsties))
        curr_trial_group_sparsity_idx = (self.num_steps - self.start_sampling_step) // 2 % len(self.trial_group_sparsties)
        do_proj = (self.num_steps - self.start_sampling_step) % 2 == 0
        curr_param_group = self.prunable_param_group_dict[self.param_group_ids[curr_param_group_idx]]
        curr_trial_group_sparsity = self.trial_group_sparsties[curr_trial_group_sparsity_idx]
        print(curr_param_group_idx, curr_trial_group_sparsity_idx, curr_param_group['id'], curr_trial_group_sparsity, do_proj)
        if do_proj:
            self.proj_trial_group_sparsity(curr_param_group, curr_trial_group_sparsity)
        else:
            # Collect loss deviation after projection
            curr_param_group['loss_collection'][self.curr_sampling_period].append(loss.item() / self.ref_loss) 
            self.reset_params()
        curr_param_group['active_violating_idxes_collection'][self.curr_sampling_period] = [i for i in range(curr_param_group['num_groups'])]          
    
    def hybrid_step(self):
        t = self.num_steps - self.cric_terminated_step - 1
        for group in self.param_groups:
            if not group['is_prunable'] or len(group['redundant_idxes']) == 0 or self.num_steps > self.cric_terminated_step + self.hybrid_steps:
                for p_name, p in zip(group['p_names'], group['params']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
            elif group['is_prunable'] and len(group['redundant_idxes']) > 0:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, group['redundant_idxes'], ...] *= (self.hybrid_steps - t - 1.0) / (self.hybrid_steps - t)
                    else:
                        p.data[group['redundant_idxes']] *= (self.hybrid_steps - t - 1.0) / (self.hybrid_steps - t)
                    
                    # Tackle auxiliary params
                    for ng_id, offset in group['auxiliary_ngs']:
                        aux_pg = self.auxiliary_param_groups[ng_id]
                        for aux_p in aux_pg['params']:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[offset:offset+group['num_groups'], ...] *= (self.hybrid_steps - t - 1.0) / (self.hybrid_steps - t)

            if self.num_steps > self.cric_terminated_step + self.hybrid_steps > 0:
                for p, p_transform in zip(group['params'], group['p_transform']):
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, group['redundant_idxes']] = 0.0
                    else:
                        p.data[group['redundant_idxes']] = 0.0
                        

    def step(self, loss=None):
        self.num_steps += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # At the very beginning of sampling process, collect loss and importance score for each node group
        if self.num_steps == self.start_sampling_step:
            self.ref_loss = loss.item() if loss is not None else None
            self.reset_cache_params()
            self.curr_sampling_period += 1 

        if self.num_steps < self.start_sampling_step:
            self.basic_step()
        elif self.num_steps >= self.start_sampling_step and self.num_steps < self.start_global_sampling_step:
            self.proj_step(loss)
        elif self.num_steps >= self.start_global_sampling_step and self.curr_sampling_period < self.sampling_periods and not self.is_cric_terminated: 
            self.cric_step()
        elif self.is_cric_terminated:
            self.hybrid_step()
        
        if not self.is_cric_terminated and self.cric_terminate():
            print("cric_terminate", self.num_steps)
            self.compute_accumulate_saliency_score()
            self.identify_redundant_groups()
            self.reset_params()
            self.is_cric_terminated = True
            self.cric_terminated_step = self.num_steps
        return

    def compute_norm_groups(self):
        norm_important_groups = 0.0
        norm_violating_groups = 0.0
        num_important_groups = 0
        num_violating_groups = 0
        num_trial_violating_groups = 0
        num_historical_violating_groups = 0

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
                violating_idxes = group['active_violating_idxes']
                import_idxes = [i for i in range(norm_group.shape[0]) if i not in violating_idxes]
                norm_important_groups += torch.sum(norm_group[import_idxes]).item()
                norm_violating_groups += torch.sum(norm_group[violating_idxes]).item()
                num_important_groups += len(import_idxes)
                num_violating_groups += len(violating_idxes)
                num_trial_violating_groups += len(group['trial_violating_idxes'])
                num_historical_violating_groups += len(group['historical_violating_idxes'])

        return norm_important_groups, norm_violating_groups, num_important_groups, num_violating_groups, \
               num_trial_violating_groups, num_historical_violating_groups

    def compute_accumulate_saliency_score(self):
        # Compute accumulated saliency score
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                param_group['accumulated_importance_score'] = None
                accumulate_count = 0
                for sample_period in param_group['importance_score_collection']:
                    importance_score_sample_steps = param_group['importance_score_collection'][sample_period]
                    for importance_score in importance_score_sample_steps:
                        if param_group['accumulated_importance_score'] is None:
                            param_group['accumulated_importance_score'] = importance_score['overall'].clone()
                        else:
                            param_group['accumulated_importance_score'] += importance_score['overall']
                        accumulate_count += 1
                if param_group['accumulated_importance_score'] is not None:
                    param_group['accumulated_importance_score'] /= float(accumulate_count)

                for sample_period in param_group['active_violating_idxes_collection']:
                    if len(param_group['active_violating_idxes_collection'][sample_period]) > 0:
                        violating_idxes = param_group['active_violating_idxes_collection'][sample_period]
                        loss_scores = param_group['loss_collection'][sample_period]
                        if len(loss_scores) == 0:
                            continue
                        # need to normalize with violating set sizes
                        avg_loss_score = sum(loss_scores) / len(loss_scores)
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
            cluster_importance_score = torch.cat(cluster_importance_scores[cluster_name], dim=0)
            target_num_redundant_groups = self.target_num_redundant_groups_by_clusters[cluster_name]

            _, top_indices = torch.topk(-cluster_importance_score, target_num_redundant_groups)
            top_indices = top_indices.cpu().numpy()

            for group in self.prunable_param_group_clusters[cluster_name]:
                if group['is_prunable'] and not group['is_auxiliary']:
                    global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])
                    group['redundant_idxes'] = (global_active_redundant_idx - group['global_start_idx']).tolist()