# MIT License
#
# Copyright (c) 2017-2019 Aleksei Tiulpin, University of Oulu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def conv_block_3x3(inp_channels, out_channels, activation='relu', normalization='BN'):
    """Standard 3x3 convolution block with normalization and activation."""
    layers = [nn.Conv2d(inp_channels, out_channels, kernel_size=3, padding=1, bias=False)]

    if normalization == 'BN':
        layers.append(nn.BatchNorm2d(out_channels))
    elif normalization == 'IN':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    elif activation == 'selu':
        layers.append(nn.SELU(inplace=True))

    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Encoder block. For encoder-decoder architecture.
    Conv3x3-Conv3x3-Maxpool 2x2. BatchNorm or other form of normalization is used optionally.
    """

    def __init__(self, inp_channels, out_channels, depth=2, activation='relu', normalization='BN'):
        super().__init__()
        layers = []
        for i in range(depth):
            in_c = inp_channels if i == 0 else out_channels
            layers.append(conv_block_3x3(in_c, out_channels, activation, normalization))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder block. For encoder-decoder architecture.
    Bilinear ups -> Conv3x3-Conv3x3. BatchNorm or other form of normalization is used optionally.
    """

    def __init__(self, inp_channels, out_channels, depth=2, mode='bilinear', activation='relu', normalization='BN'):
        super().__init__()
        self.ups_mode = mode
        layers = []
        for i in range(depth):
            in_c = inp_channels if i == 0 else out_channels
            layers.append(conv_block_3x3(in_c, out_channels, activation, normalization))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.shape[-2:], mode=self.ups_mode, align_corners=True)
        y = torch.cat([x_ups, x_big], dim=1)
        return self.layers(y)


class UNet(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on depth and width.

    Parameters
    ----------
    bw : int
        Basic width of the network, which is doubled at each layer.
    depth : int
        Number of downsampling/upsampling layers.
    center_depth : int
        Depth of the central block in UNet.
    n_inputs : int
        Number of input channels.
    n_classes : int
        Number of output segmentation classes.
    activation : str
        Activation function ('relu', 'selu', 'elu').
    norm_encoder : str
        Normalization for encoder blocks ('BN', 'IN', None).
    norm_decoder : str
        Normalization for decoder blocks ('BN', 'IN', None).
    norm_center : str
        Normalization for the central block ('BN', 'IN', None).
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        bw=24,
        depth=6,
        center_depth=2,
        activation='relu',
        norm_encoder='BN',
        norm_decoder='BN',
        norm_center='BN',
        pretrained_model=False,
        path_weights='',
        path_pretrained='',
        restore_weights=False,
    ):
        super().__init__()
        self.depth = depth

        self.downs = nn.ModuleDict()
        self.downs['down1'] = Encoder(input_channels, bw, activation=activation, normalization=norm_encoder)

        mul_out = 1
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            self.downs[f'down{level}'] = Encoder(
                bw * mul_in, bw * mul_out, activation=activation, normalization=norm_encoder
            )

        self.center = nn.Sequential(
            *[conv_block_3x3(bw * mul_out, bw * mul_out, activation, norm_center) for _ in range(center_depth)]
        )

        self.ups = nn.ModuleDict()
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            self.ups[f'up{level}'] = Decoder(
                2 * bw * mul_in, bw * mul_in // 2, activation=activation, normalization=norm_decoder
            )

        self.ups['up1'] = Decoder(2 * bw, bw, activation=activation, normalization=norm_decoder)
        self.mixer = nn.Conv2d(bw, output_channels, kernel_size=1, padding=0, stride=1, bias=True)

        # Handle weight restoration
        target_path = None
        if restore_weights and path_weights and os.path.isfile(path_weights):
            target_path = path_weights
        elif pretrained_model and path_pretrained and os.path.isfile(path_pretrained):
            target_path = path_pretrained

        if target_path:
            state_dict = torch.load(target_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict)

    def forward(self, x):
        encoded_results = {}
        out = x

        for level in range(1, self.depth + 1):
            key = f'down{level}'
            convolved, pooled = self.downs[key](out)
            encoded_results[key] = convolved
            out = pooled

        out = self.center(out)

        for level in reversed(range(1, self.depth + 1)):
            out = self.ups[f'up{level}'](encoded_results[f'down{level}'], out)

        return self.mixer(out)


if __name__ == '__main__':
    # Quick sanity check
    model = UNet(bw=16, depth=4, n_inputs=1, n_classes=2)
    dummy_input = torch.randn(2, 1, 128, 128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")