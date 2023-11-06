import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models import vgg19, VGG19_Weights, vgg11, VGG11_Weights, resnet18, ResNet18_Weights, resnet50, \
    ResNet50_Weights


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


# Define the Upsample Block
class UpsampleBLock(nn.Module):
    def __init__(self, in_features, upscale_factor):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_features, in_features * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# Define the Generator
class Generator(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Generator, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsample 2x for each block
        self.upsample = nn.Sequential(
            UpsampleBLock(64, 2),
            UpsampleBLock(64, 2)
        )

        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.input(x)
        x = self.residuals(initial)
        x = self.output(x) + initial
        x = self.upsample(x)
        x = self.final(x)
        return x




# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

def calculate_psnr(sr, hr, device):
    mse = torch.mean((sr - hr) ** 2).to(device)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).to(device)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load the pre-trained ResNet-18 model
        resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer (fc) to get the features before classification
        self.feature_extractor = torch.nn.Sequential(*list(resnet50_model.children())[:-1])

    def forward(self, img):
        # Normalize the img to the ImageNet distribution before passing it through the feature extractor
        # The normalization values are the ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        # img = interpolate(img, mode='bilinear', size=(224, 224), align_corners=False)

        # Forward pass through the feature extractor
        features = self.feature_extractor(img)
        # Flatten the features to a single vector per image
        return torch.flatten(features, 1)