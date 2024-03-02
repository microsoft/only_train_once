import torch
from only_train_once import OTO
from backends import DemonetBatchnormPruning
import unittest
import os
import torch.nn as nn

OUT_DIR = './cache'

class TestDemoNetBatchnormPruningCase1(unittest.TestCase):
    def test_sanity(self):
        model = DemonetBatchnormPruning(13,32,256,5,3,nn.LeakyReLU(),False,256)
        dummy_input=[torch.rand(1, 3, 256, 256),torch.rand(1, 4, 256, 256),torch.rand(1, 6, 256, 256)]
        oto = OTO(model, dummy_input)
        node_groups = oto._graph.node_groups
        
        skip_strs = ['decoder']
        for skip_str in skip_strs:
            for key in node_groups:
                node_group = node_groups[key]
                for node_name in node_group.nodes:
                    for str_name in node_group.nodes[node_name].param_names:
                        if skip_str in str_name:
                            node_group.is_prunable = False
                            break
        
        oto.visualize(view=False, out_dir=OUT_DIR)
        # oto.random_set_zero_groups(target_group_sparsity=0.5)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")
        self.assertLessEqual(max_output_diff, 1e-4)