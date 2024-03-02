from .tensor_transform import TensorTransform

def index_transformation_param_group(indexes_by_groups, transformation_type, param_group):
    if transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        return index_transformation(indexes_by_groups, transformation_type, num_heads=param_group['num_heads'],  head_dim=param_group['head_dim'])
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD or transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        return index_transformation(indexes_by_groups, transformation_type, head_dim=param_group['head_dim'])
    else:
        return index_transformation(indexes_by_groups, transformation_type)

def index_transformation(indexes_by_groups, transformation_type, num_heads=1, head_dim=1):
    if transformation_type == TensorTransform.NO_UPDATE or \
       transformation_type == TensorTransform.NO_PRUNE or \
       transformation_type == TensorTransform.BASIC or \
       transformation_type == TensorTransform.ACCESSORY or \
       transformation_type == TensorTransform.TRANSPOSE:
        return indexes_by_groups 
    elif transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        refined_indexes = [i for i in indexes_by_groups]
        for h in range(1, num_heads):
            refined_indexes.extend([i + head_dim * h for i in indexes_by_groups])
        return refined_indexes
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD or \
         transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        refined_indexes = list()
        for i in indexes_by_groups:
            refined_indexes.extend([h + i * head_dim for h in range(head_dim)])
        return refined_indexes
    else:
        return indexes_by_groups