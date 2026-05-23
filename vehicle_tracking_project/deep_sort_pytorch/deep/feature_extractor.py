
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out))
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out))
            self.is_downsample = True

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.is_downsample:
            x = self.downsample(x)
        return self.relu(x + y)

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    layers = [BasicBlock(c_in, c_out, is_downsample=is_downsample)]
    for _ in range(1, repeat_times):
        layers.append(BasicBlock(c_out, c_out))
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super().__init__()
        self.conv    = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ELU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1  = make_layers(32,  32, 2, False)
        self.layer2  = make_layers(32,  64, 2, True)
        self.layer3  = make_layers(64, 128, 2, True)
        self.dense   = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ELU(inplace=True))
        self.reid    = reid
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=-1).mean(dim=-1)   # global avg pool
        x = self.dense(x)
        if self.reid:
            x = x / x.norm(p=2, dim=1, keepdim=True)
            return x
        return self.classifier(x)

class Extractor:
    def __init__(self, model_path, use_cuda=True):
        self.net    = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict  = torch.load(model_path, map_location=torch.device("cpu"))
        # strip "module." prefix if present (DataParallel checkpoint)
        state_dict  = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.device).eval()
        self.size   = (64, 128)
        self.norm   = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _preprocess(self, im_crops):
        imgs = []
        for im in im_crops:
            img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            imgs.append(self.norm(img))
        return torch.stack(imgs, dim=0).float()

    def __call__(self, im_crops):
        if len(im_crops) == 0:
            return np.array([])
        imgs = self._preprocess(im_crops).to(self.device)
        with torch.no_grad():
            features = self.net(imgs)
        return features.cpu().numpy()
