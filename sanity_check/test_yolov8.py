import torch
from only_train_once import OTO
import torchvision.models
import unittest
import os
from ultralytics import YOLO
        
OUT_DIR = './cache'

class TestYolov8(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):


        # Load a model
        # model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        # model.train(data="coco128.yaml", epochs=3)  # train the model
        # metrics = model.val()  # evaluate model performance on the validation set
        # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        path = model.export(format="onnx")  # export the model to ONNX format
        exit()
        oto = OTO(model, dummy_input)
        oto.visualize(view=False, out_dir=OUT_DIR)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        