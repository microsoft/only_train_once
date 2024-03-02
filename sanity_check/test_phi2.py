import torch
from only_train_once import OTO
import unittest
import os
from backends import PhiConfig, PhiForCausalLM
from transformers import AutoTokenizer

OUT_DIR = './cache'

class TestPhi2(unittest.TestCase):
    def test_sanity(self, dummy_input=None):
        phi_config = PhiConfig()
        phi_config.n_layer = 20
        model = PhiForCausalLM(phi_config)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        text = 'This is a test sentence of a very long string and random wording that is used to test dolly model.' * 7
        input_data = tokenizer(text, return_tensors='pt', return_attention_mask=False).input_ids

        oto = OTO(model, dummy_input=(input_data,), strict_out_nodes=True)
        oto.mark_unprunable_by_node_ids(['node-215'])
        oto.visualize(view=False, out_dir=OUT_DIR)
        
        oto.random_set_zero_groups()

        oto.construct_subnet(
            export_huggingface_format=False,
            export_float16=False,
            full_group_sparse_model_dir=OUT_DIR,
            compressed_model_dir=OUT_DIR
        )

        text_1 = 'This is a test sentence of a very long string and random wording that is used to test dolly model.' * 7
        input_data_1 = tokenizer(text_1, return_tensors='pt').input_ids

        text_2 = 'This is a good test sentence of a pretty short string and wording that is used to test dolly model.' * 7
        input_data_2 = tokenizer(text_2, return_tensors='pt').input_ids
        
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)
        full_output_1 = full_model(input_data_1.to(full_model.device))
        full_output_2 = full_model(input_data_2.to(full_model.device))
        compressed_output_1 = compressed_model(input_data_1.to(compressed_model.device))
        compressed_output_2 = compressed_model(input_data_2.to(compressed_model.device))
        max_output_diff_1 = torch.max(full_output_1.logits - compressed_output_1.logits).item()
        max_output_diff_2 = torch.max(full_output_2.logits - compressed_output_2.logits).item()
        max_output_diff_3 = torch.max(full_output_1.logits - compressed_output_2.logits).item()
        max_output_diff_4 = torch.max(full_output_2.logits - compressed_output_1.logits).item()
        
        print("Maximum output difference under the same inputs: ", max_output_diff_1)
        print("Maximum output difference under the same inputs: ", max_output_diff_2)
        print("Maximum output difference under the different inputs: ", max_output_diff_3)
        print("Maximum output difference under the different inputs: ", max_output_diff_4)

        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        self.assertLessEqual(max_output_diff_1, 3.0)
        self.assertLessEqual(max_output_diff_2, 3.0)
        self.assertLessEqual(max_output_diff_3, 6.0)
        self.assertLessEqual(max_output_diff_4, 6.0)