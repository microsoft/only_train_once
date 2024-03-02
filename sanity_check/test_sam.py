import unittest
# from PIL import Image
import torch
# import requests

# from transformers import SamProcessor
from transformers import SamConfig

from backends import SamModel
from only_train_once import OTO

OUT_DIR = './cache'

class TestSam(unittest.TestCase):

    def test_sam_model(self):
        dummy_input = torch.randn(1, 3, 1024,1024)
        config = SamConfig()
        # config.vision_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_multimask_outputs = 1
        model = SamModel(config)

        # model = SamModel.from_pretrained("facebook/sam-vit-base")
        # print(config)
        # exit()
        # for name, param in model.named_parameters():
        #     print(name, param.shape)
        # exit()
        # torch.onnx.export(
        #     model.vision_encoder, 
        #     torch.randn(1, 3, 1024,1024),
        #     'sam_1layers_encoder.onnx'
        # )
        # exit()
        oto = OTO(model.vision_encoder, dummy_input)
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
         # For test FLOP and param reductions. 
        # full_flops = oto.compute_flops(in_million=True)['total']
        # full_num_params = oto.compute_num_params(in_million=True)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)


if __name__ == '__main__':
    unittest.main()