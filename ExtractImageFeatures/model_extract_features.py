"""
特徴抽出に用いるモデルを定義
author: Kouki Nishimoto, Yuma Takeda
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ConvNextFeatureExtractor, ConvNextModel

resnet50_model_path = "/home/kouki/Models/resnet50-19c8e357.pth"

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
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


class ConvNextEncoder():
  def __init__(self):
    self.feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-224-22k-1k")
    self.model = ConvNextModel.from_pretrained("facebook/convnext-xlarge-224-22k-1k")

  # 引数imgを2048次元の特徴量にして返す．
  def encode(self, img):
    inputs = self.feature_extractor(img, return_tensors="pt")
    with torch.no_grad():
      outputs = self.model(**inputs)
    return outputs.last_hidden_state