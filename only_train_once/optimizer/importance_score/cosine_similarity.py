import torch
import torch.nn.functional as F
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

def importance_score_by_cosine_similarity(param_group):
    norm_params = None
    norm_grads = None
    params_grads_inner_prod = None
    # for param, grad, p_transform in zip(param_group['params'], param_group['grad_variant'], param_group['p_transform']):
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):    
        if p_name not in param_group['grad_variant']:
            continue
        grad = param_group['grad_variant'][p_name]
        param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        if norm_params == None:
            norm_params = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_params += torch.norm(param_transform, dim=1) ** 2

        grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'])
        if norm_grads == None:
            norm_grads = torch.norm(grad_transform, dim=1) ** 2
        else:
            norm_grads += torch.norm(grad_transform, dim=1) ** 2
        
        if params_grads_inner_prod == None:
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)
    param_group['importance_scores']['cosine_similarity'] = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8) + 1

def importance_score_by_cosine_similarity_lora(param_group, global_params):
    norm_params = None
    norm_grads = None
    params_grads_inner_prod = None
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + 'weight'
                original_param = global_params[original_param_name]

                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'], param_group['num_heads'])
                elif lora_strs[0] == 'lora_embedding_B':
                    param_transform = tensor_transformation(original_param, TensorTransform.TRANSPOSE, param_group['num_groups'])
                else:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'])

                if norm_params == None:
                    norm_params = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_params += torch.norm(param_transform, dim=1) ** 2

                grad_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'], param_group['num_heads'])
                else:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'])
                if norm_grads == None:
                    norm_grads = torch.norm(grad_transform, dim=1) ** 2
                else:
                    norm_grads += torch.norm(grad_transform, dim=1) ** 2

                if params_grads_inner_prod == None:
                    params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
                else:
                    params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)
    param_group['importance_scores']['cosine_similarity'] = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8) + 1    
