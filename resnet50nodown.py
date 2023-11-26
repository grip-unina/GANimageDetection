# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II
# of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
"""
The `resnet50nodown` module implements a variant of the ResNet-50 architecture without downsampling.

This module provides the ResNet-50 model adapted for tasks where maintaining the spatial resolution
of input images is crucial. It modifies the traditional ResNet-50 architecture by avoiding 
downsampling in certain layers, making it suitable for applications like image segmentation or 
high-resolution image analysis.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

LIMIT_SIZE = 1536
LIMIT_SLIDE = 1024


class ChannelLinear(nn.Linear):
    """
    A linear transformation layer that operates over channels of the input tensor.

    This class is a specialized form of the PyTorch Linear layer where the linear
    transformation is applied across the channel dimension of the input tensor.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initializes the ChannelLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If set to `False`, the layer will not learn an additive bias.
                         Default is `True`.
        """
        super(ChannelLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        """
        Defines the computation performed at every call of the layer.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                        C is the number of channels, H is the height, and W is the width.

        Returns:
            Tensor: Output tensor after applying the linear transformation.
        """
        out_shape = [x.shape[0], x.shape[2], x.shape[3], self.out_features]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.in_features)
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = x.view(out_shape).permute(0, 3, 1, 2)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """
    Creates a 3x3 convolution layer with padding and specified number of input and output planes.

    Args:
        in_planes (int): Number of input planes (channels).
        out_planes (int): Number of output planes (channels).
        stride (int): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer with the specified parameters.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    Creates a 1x1 convolution layer with the specified number of input and output planes.

    Args:
        in_planes (int): Number of input planes (channels).
        out_planes (int): Number of output planes (channels).
        stride (int): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer with the specified parameters.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck architecture for ResNet. Creates a deeper residual function with a bottleneck design.

    The bottleneck design consists of three layers: a 1x1 convolution which reduces the dimension,
    a 3x3 convolution, and another 1x1 convolution that restores the dimension. The bottleneck block
    improves the efficiency of the network.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes the Bottleneck layer.

        Args:
            inplanes (int): Number of input planes (channels).
            planes (int): Intermediate number of planes (channels) in the bottleneck.
            stride (int): Stride for the second convolutional layer. Default is 1.
            downsample (nn.Module, optional): Downsampling layer if needed. Default is None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Defines the forward pass of the Bottleneck layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the Bottleneck layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Implements a Residual Network (ResNet) without downsampling.

    This class represents a variant of the ResNet architecture which avoids downsampling.
    It consists of several residual blocks (Bottlenecks) that enable the network to learn
    complex features without losing spatial resolution.
    """

    def __init__(self, block, layers, num_classes=1, stride0=2):
        """
        Initializes the ResNet model.

        Args:
            block (nn.Module): The block type to be used, e.g., Bottleneck.
            layers (list of int): Number of layers for each block of the network.
            num_classes (int): Number of classes for the final output. Default is 1.
            stride0 (int): Stride for the first layer of the network. Default is 2.
        """
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=stride0, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transform form Pillow
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Creates a sequential layer block for the ResNet model.

        Args:
            block (nn.Module): The block type to be used in the layer (e.g., Bottleneck).
            planes (int): Number of planes (channels) in the bottleneck structure.
            blocks (int): Number of blocks to be created in this layer.
            stride (int): Stride for the convolutional layer. Default is 1.

        Returns:
            nn.Sequential: A sequential layer consisting of specified blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def change_output(self, num_classes):
        """
        Changes the output dimension of the final fully connected layer.

        Args:
            num_classes (int): The number of classes for classification.

        Returns:
            ResNet: The modified ResNet instance with updated output layer.
        """
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self

    def change_input(self, num_inputs):
        """
        Adjusts the first convolutional layer to accept a different number of input channels.

        Args:
            num_inputs (int): The new number of input channels.

        Returns:
            ResNet: The modified ResNet instance with updated input layer.
        """
        data = self.conv1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs > old_num_inputs:
            times = num_inputs // old_num_inputs
            if (times * old_num_inputs) < num_inputs:
                times = times + 1
            data = data.repeat(1, times, 1, 1) / times
        elif num_inputs == old_num_inputs:
            return self

        data = data[:, :num_inputs, :, :]
        print(self.conv1.weight.data.shape, "->", data.shape)
        self.conv1.weight.data = data

        return self

    def feature(self, x):
        """
        Extracts feature maps from the input tensor.

        This method applies a series of convolutional and pooling layers to the input tensor
        to extract high-level features.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The feature maps extracted from the input.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.

        Args:
            x (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor of the network after passing through all layers.
        """
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def apply(self, pil):
        device = self.conv1.weight.device
        if (pil.size[0] > LIMIT_SIZE) and (pil.size[1] > LIMIT_SIZE):
            print("err:", pil.size)
            with torch.no_grad():
                img = self.transform(pil)
                list_logit = list()
                list_weight = list()
                for index0 in range(0, img.shape[-2], LIMIT_SLIDE):
                    for index1 in range(0, img.shape[-1], LIMIT_SLIDE):
                        clip = img[
                            ...,
                            index0 : min(index0 + LIMIT_SLIDE, img.shape[-2]),
                            index1 : min(index1 + LIMIT_SLIDE, img.shape[-1]),
                        ]
                        logit = (
                            torch.squeeze(self(clip.to(device)[None, :, :, :]))
                            .cpu()
                            .numpy()
                        )
                        weight = clip.shape[-2] * clip.shape[-1]
                        list_logit.append(logit)
                        list_weight.append(weight)

            logit = np.mean(np.asarray(list_logit) * np.asarray(list_weight)) / np.mean(
                list_weight
            )
        else:
            with torch.no_grad():
                logit = (
                    torch.squeeze(self(self.transform(pil).to(device)[None, :, :, :]))
                    .cpu()
                    .numpy()
                )

        return logit


def resnet50nodown(device, filename, num_classes=1):
    """
    Constructs and returns a ResNet-50 model without downsampling.

    Args:
        device (torch.device): The device to load the model onto.
        filename (str): Path to the file containing the model weights.
        num_classes (int): Number of classes for the final output. Default is 1.

    Returns:
        ResNet: The ResNet-50 model loaded with the specified weights.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stride0=1)
    model.load_state_dict(
        torch.load(filename, map_location=torch.device("cpu"))["model"]
    )
    model = model.to(device).eval()
    return model
