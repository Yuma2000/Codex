"""
特徴抽出に用いるモデルを定義
author: Kouki Nishimoto, Yuma Takeda
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Import section for ConvNeXt --- #
from transformers import ConvNextFeatureExtractor, ConvNextModel
from datasets import load_dataset

# --- Variable declaration section --- #
resnet50_model_path = "/home/kouki/Models/resnet50-19c8e357.pth"

# --- ResNet50 Model section --- #
class I2FEncoder(nn.Module):
    def __init__(self):
        super(I2FEncoder, self).__init__()
        self.resnet50 = models.resnet50()
        self.resnet50.load_state_dict(torch.load(resnet50_model_path))
        del self.resnet50.fc
        
    #転移学習
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# --- ConvNeXt Model section --- #
class ConvNextEncoder():
    def __init__(self):
        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]

        feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-224-22k-1k")
        model = ConvNextModel.from_pretrained("facebook/convnext-xlarge-224-22k-1k")
        inputs = feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state