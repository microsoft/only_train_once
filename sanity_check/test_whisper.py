import unittest
from PIL import Image
import torch
import requests
from transformers import WhisperProcessor
from backends import WhisperForConditionalGeneration
from only_train_once import OTO

from datasets import load_dataset

OUT_DIR = './cache'

class TestCode(unittest.TestCase):

    def test_whisper_model(self):
        scale=2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = ds[0]["audio"]
        inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
        oto = OTO(model, (inputs, scale))
        oto.visualize(view=False, out_dir=OUT_DIR)
         # For test FLOP and param reductions. 
        full_flops = oto.compute_flops(in_million=True)['total']
        full_num_params = oto.compute_num_params(in_million=True)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)

if __name__ == '__main__':
    unittest.main()