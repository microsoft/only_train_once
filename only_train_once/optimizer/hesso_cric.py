import torch
import numpy as np
from torch.optim.optimizer import required
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation_param_group

class HESSOCRIC(BaseHybridSparseOptimizer):
    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.5, tolerance=0, group_divisible=1, \
                 start_cric_step=0, max_cycle_period=10, sampling_steps=None, hybrid_training_steps=None, \
                 importance_score_criteria='default'):

        print("Setup HESSOCRIC")
        self.start_cric_step = start_cric_step
        self.sampling_steps = sampling_steps # How many sampling step for each cric cycle
        self.max_cycle_period = int(max(1, max_cycle_period)) # How many periods that the pruning last for.
        self.curr_cycle_period = -1 # Track pruning periodp
        self.hybrid_training_steps = hybrid_training_steps

        self.redundant_groups_identified = False
        self.tolerance = tolerance

        if importance_score_criteria == 'default' or importance_score_criteria is None:
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2,\
                                              'cosine_similarity': 0.2, \
                                              'taylor_first_order': 0.2, 'taylor_second_order': 0.2, 'loss': 1.0}
        else:
            self.importance_score_criteria = importance_score_criteria

        super(HESSOCRIC, self).__init__(params=params, variant=variant, lr=lr, first_momentum=first_momentum, second_momentum=second_momentum, \
                                        dampening=dampening, weight_decay=weight_decay, target_group_sparsity=target_group_sparsity, \
                                        group_divisible=group_divisible)

        for param_group in self.param_groups:
            param_group['important_idxes'] = [i for i in range(param_group['num_groups'])]
            param_group['active_violating_idxes'] = list()
            param_group['trial_violating_idxes'] = list()
            param_group['historical_violating_idxes'] = list()
            param_group['active_redundant_idxes'] = list()
            param_group['pruned_idxes'] = list()
            param_group['importance_scores'] = dict()
            
        print("Total number of groups")
        print(self.target_group_sparsity, self.total_num_groups, self.target_num_redundant_groups)

        # Create param dictionary for facilitating accessing lora_A modules
        self.cache_parameters = dict()
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.cache_parameters[p_name] = param.data.clone()        
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                param_group['importance_score_collection'] = dict()
                param_group['active_violating_idxes_collection'] = dict()
                param_group['loss_collection'] = dict()
                for cycle_period in range(self.max_cycle_period + 1):
                    param_group['importance_score_collection'][cycle_period] = list()
                    param_group['active_violating_idxes_collection'][cycle_period] = list()
                    param_group['loss_collection'][cycle_period] = list()

        self.prunable_param_group_dict = dict()
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                self.prunable_param_group_dict[param_group['id']] = param_group
        self.param_group_ids = list(self.prunable_param_group_dict.keys())
        self.trial_group_sparsties = [0.25, 0.5, 0.75]
        self.start_global_sampling_step = 2 * len(self.trial_group_sparsties) * len(self.prunable_param_group_dict) + self.start_cric_step

        self.is_cric_terminated = False

    def reset_cache_params(self):
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.cache_parameters[p_name] = param.data.clone()  

    def reset_params(self):
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                param.data.copy_(self.cache_parameters[p_name]) # ?
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()

    def compute_num_active_violating_groups(self):
        num_violating_groups = 0
        for param_group in self.param_groups:
            num_violating_groups += len(param_group['active_violating_idxes'])
        return num_violating_groups

    def cric_terminate(self):
        if self.curr_cycle_period >= self.max_cycle_period:
            return True
        elif self.curr_cycle_period >= 1 and self.compute_num_active_violating_groups() <= self.tolerance:
            return True
        else:
            return False

    def update_violating_set(self, cycle_period=0):
        print("Update violating set")
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_K = self.target_num_redundant_groups

        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy().tolist()
    
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                if cycle_period == 1:
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    group['active_violating_idxes'] = (global_trial_violating_idxes - group['global_start_idx']).tolist()
                else:
                    group['active_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['historical_violating_idxes']]
                group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['active_violating_idxes']]

    def update_trial_violating_set(self):
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_K = self.target_num_redundant_groups

        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy().tolist()

        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                group['trial_violating_idxes'] += (global_trial_violating_idxes - group['global_start_idx']).tolist()
                group['trial_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['active_violating_idxes'] \
                                                  and i not in group['historical_violating_idxes']]
                group['trial_violating_idxes'] = list(set(group['trial_violating_idxes']))

    def update_historical_violating_set(self):
        for param_group in self.param_groups:
            param_group['historical_violating_idxes'] += param_group['active_violating_idxes']
            param_group['historical_violating_idxes'] = list(set(param_group['historical_violating_idxes']))

    def proj_trial_group_sparsity(self, param_group, trial_group_sparsity):
        if 'importance_scores' not in param_group:
            return

        num_redund_grps = max(min(int(param_group['num_groups'] * trial_group_sparsity), param_group['num_groups']), 1)
        _, proj_indices = torch.topk(-param_group['importance_scores']['overall'], num_redund_grps)
        proj_indices = proj_indices.cpu().numpy().tolist()
        for (p, p_transform) in zip(param_group['params'], param_group['p_transform']):
            proj_indices = index_transformation_param_group(proj_indices, p_transform, param_group)
            if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                p.data[:, proj_indices, ...] = 0.0
            else:
                p.data[proj_indices] = 0.0

    def cric_step(self):
        # print("Current sampling period", self.curr_sampling_period, self.num_steps)
        self.compute_importance_scores()
        self.commit_important_scores(self.curr_cycle_period)

        if (self.num_steps - self.start_global_sampling_step) % self.sampling_steps == 0:
            self.curr_cycle_period += 1
            self.update_violating_set(self.curr_cycle_period)
            self.update_historical_violating_set()
            self.reset_params()
        self.update_trial_violating_set()
        
        # Second pass to update variables
        t = (self.num_steps - self.start_global_sampling_step - 1) % self.sampling_steps
        for group in self.param_groups:
            if not (group['is_prunable'] and not group['is_auxiliary']):
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
        for param_group in self.param_groups:
            self.gradient_descent_step(param_group)

    def proj_step(self, loss):
        self.compute_importance_scores()
        self.commit_important_scores(self.curr_cycle_period)

        curr_param_group_idx = (self.num_steps - self.start_cric_step) // (2 * len(self.trial_group_sparsties))
        curr_trial_group_sparsity_idx = (self.num_steps - self.start_cric_step) // 2 % len(self.trial_group_sparsties)
        do_proj = (self.num_steps - self.start_cric_step) % 2 == 0
        curr_param_group = self.prunable_param_group_dict[self.param_group_ids[curr_param_group_idx]]
        curr_trial_group_sparsity = self.trial_group_sparsties[curr_trial_group_sparsity_idx]

        if do_proj:
            self.proj_trial_group_sparsity(curr_param_group, curr_trial_group_sparsity)
        else:
            # Collect loss deviation after projection
            curr_param_group['loss_collection'][self.curr_cycle_period].append(loss.item() / self.ref_loss) 
            self.reset_params()
        curr_param_group['active_violating_idxes_collection'][self.curr_cycle_period] = [i for i in range(curr_param_group['num_groups'])]          

    def hybrid_step(self):
        t = self.num_steps - self.cric_terminated_step - 1
        for group in self.param_groups:
            if not group['is_prunable'] or len(group['active_redundant_idxes']) == 0 or self.num_steps > self.cric_terminated_step + self.hybrid_training_steps:
                for p_name, p in zip(group['p_names'], group['params']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
            elif group['is_prunable'] and len(group['active_redundant_idxes']) > 0:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])

                    active_redundant_idxes = index_transformation_param_group(group['active_redundant_idxes'], p_transform, group)
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, active_redundant_idxes, ...] *= (self.hybrid_training_steps - t - 1.0) / (self.hybrid_training_steps - t)
                    else:
                        p.data[active_redundant_idxes] *= (self.hybrid_training_steps - t - 1.0) / (self.hybrid_training_steps - t)
                    
                    # Tackle auxiliary params
                    for ng_id, offset in group['auxiliary_ngs']:
                        active_redundant_aux_idxes = [i + offset for i in active_redundant_idxes]
                        for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[active_redundant_aux_idxes, ...] *= (self.hybrid_training_steps - t - 1.0) / (self.hybrid_training_steps - t)

            if self.num_steps == self.cric_terminated_step + self.hybrid_training_steps:
                group['pruned_idxes'].extend(group['active_redundant_idxes'])
                group['active_redundant_idxes'].clear()

            if self.num_steps > self.cric_terminated_step + self.hybrid_training_steps:
                self.fix_pruned_groups_as_zeros(group)

    def step(self, loss=None, closure=None):
        if closure is not None:
            loss = closure()

        self.num_steps += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # At the very beginning of sampling process, collect loss and importance score for each node group
        if self.num_steps == self.start_cric_step:
            self.ref_loss = loss.item() if loss is not None else None
            self.reset_cache_params()
            self.curr_cycle_period += 1 

        if self.num_steps < self.start_cric_step:
            self.basic_step()
        elif self.num_steps >= self.start_cric_step and self.num_steps < self.start_global_sampling_step:
            self.proj_step(loss)
        elif self.num_steps >= self.start_global_sampling_step and self.curr_cycle_period < self.max_cycle_period and not self.is_cric_terminated: 
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
            # TODO: Remove redundant information for saving memory
        return

    def compute_accumulate_saliency_score(self):
        # Compute accumulated saliency score
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                param_group['accumulated_importance_score'] = None
                accumulate_count = 0
                for cycle_period in param_group['importance_score_collection']:
                    importance_score_sample_steps = param_group['importance_score_collection'][cycle_period]
                    for importance_score in importance_score_sample_steps:
                        if param_group['accumulated_importance_score'] is None:
                            param_group['accumulated_importance_score'] = importance_score['overall'].clone()
                        else:
                            param_group['accumulated_importance_score'] += importance_score['overall']
                        accumulate_count += 1
                if param_group['accumulated_importance_score'] is not None:
                    param_group['accumulated_importance_score'] /= float(accumulate_count)

                for cycle_period in param_group['active_violating_idxes_collection']:
                    if len(param_group['active_violating_idxes_collection'][cycle_period]) > 0:
                        violating_idxes = param_group['active_violating_idxes_collection'][cycle_period]
                        loss_scores = param_group['loss_collection'][cycle_period]
                        if len(loss_scores) == 0:
                            continue
                        # need to normalize with violating set sizes
                        avg_loss_score = sum(loss_scores) / len(loss_scores) / len(violating_idxes)
                        param_group['accumulated_importance_score'][violating_idxes] += self.importance_score_criteria['loss'] * avg_loss_score

    def identify_redundant_groups(self):
        accumulated_global_scores = list()
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                accumulated_global_scores.append(group['accumulated_importance_score'])
        
        accumulated_global_scores = torch.cat(accumulated_global_scores, dim=0)
        _, top_indices = torch.topk(-accumulated_global_scores, self.target_num_redundant_groups)
        top_indices = top_indices.cpu().numpy()

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

    def commit_important_scores(self, cycle_period):
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['importance_score_collection'][cycle_period].append(group['importance_scores'])

    def compute_metrics(self):
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.norm_violating_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0
        self.opt_metrics.num_violating_groups = 0
        self.opt_metrics.num_trial_violating_groups = 0
        self.opt_metrics.num_historical_violating_groups = 0
        
        for group in self.param_groups:
            if not (group['is_prunable'] and not group['is_auxiliary']):
                continue
            norm_group = None
            import_idxes = group['important_idxes']
            redund_idxes = group['active_redundant_idxes'] + group['pruned_idxes']
            violat_idxes = group['active_violating_idxes']
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
            self.opt_metrics.norm_violating_groups += torch.sum(norm_group[violat_idxes]).item()
            self.opt_metrics.num_important_groups += len(import_idxes)
            self.opt_metrics.num_redundant_groups += len(redund_idxes)
            self.opt_metrics.num_violating_groups += len(violat_idxes)
            self.opt_metrics.num_trial_violating_groups += len(group['trial_violating_idxes'])
            self.opt_metrics.num_historical_violating_groups += len(group['historical_violating_idxes'])

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(self.total_num_groups + self.safe_guard)

        return self.opt_metrics

