import logging
from typing import Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

def weights_init(module: nn.Module):
    """
     This is used to initialize weights of any network
    :param module: Module object that it's weight should be initialized
    :return:
    """
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.xavier_normal_(module.weight, 0.01)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0)
    elif class_name.find("nn.BatchNorm2d") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

    elif class_name.find("LocalNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

class LocalNorm(nn.Module):
    """
    Local Normalization class
    """

    def __init__(self, num_features: int):
        """
        Init
        :param num_features: Number of features
        """
        super(LocalNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.get_local_mean = nn.AvgPool2d(33, 1, 16, count_include_pad=False)

        self.get_var = nn.AvgPool2d(33, 1, 16, count_include_pad=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward run
        :param input_tensor: The input tensor
        :return: The normalized tenosr
        """
        local_mean = self.get_local_mean(input_tensor)
        # print(local_mean)
        centered_input_tensor = input_tensor - local_mean
        # print(centered_input_tensor)
        squared_diff = centered_input_tensor ** 2
        # print(squared_diff)
        local_std = self.get_var(squared_diff) ** 0.5
        # print(local_std)
        normalized_tensor = centered_input_tensor / (local_std + 1e-8)

        return normalized_tensor  # * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss
    """

    def __init__(self, use_l1: Optional[bool] = False):
        """
        Init
        :param use_l1: Flag for use l1 loss or not
        """
        super(WeightedMSELoss, self).__init__()

        self.unweighted_loss = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        Feed forward run
        :param input_tensor: The input tensor
        :param target_tensor: The output tensor
        :param loss_mask: The loss mask tensor
        :return: The Weighted loss
        """
        if loss_mask is not None:
            e = (target_tensor.detach() - input_tensor) ** 2
            e *= loss_mask
            return torch.sum(e) / torch.sum(loss_mask)
        else:
            return self.unweighted_loss(input_tensor, target_tensor)

class Unet(nn.Module):
    """ Architecture of the Unet, uses res-blocks """

    def __init__(
            self,
            n_mels: Optional[int] = 64,
            n_blocks: Optional[int] = 6,
            n_downsampling: Optional[int] = 3,
            use_bias: Optional[bool] = True,
            skip_flag: Optional[bool] = True,
    ):
        """
        Init
        :param n_mels: The base number of channels
        :param n_blocks: The number of res blocks
        :param n_downsampling: The number of downsampling blocks
        :param use_bias: Use bias or not
        :param skip_flag: Use skip connections or not
        """
        super(Unet, self).__init__()

        # Determine whether to use skip connections
        self.skip = skip_flag

        # Entry block
        # First conv-block, no stride so image dims are kept and channels dim is expanded (pad-conv-norm-relu)
        self.entry_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(1, n_mels, kernel_size=7, bias=use_bias)
            ),
            nn.BatchNorm2d(n_mels),
            nn.LeakyReLU(0.2, True),
        )

        # Downscaling
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        self.downscale_block = RescaleBlock(n_downsampling, 0.5, n_mels, True)

        # Bottleneck
        # A sequence of res-blocks
        bottleneck_block = []
        for _ in range(n_blocks):
            # noinspection PyUnboundLocalVariable
            bottleneck_block += [
                ResnetBlock(n_mels * 2 ** n_downsampling, use_bias=use_bias)
            ]
        self.bottleneck_block = nn.Sequential(*bottleneck_block)

        # Upscaling
        # A sequence of transposed-conv-blocks, Image dims expand by 2, channels dim shrinks by 2 at each block\
        self.upscale_block = RescaleBlock(n_downsampling, 2.0, n_mels, True)

        # Final block
        # No stride so image dims are kept and channels dim shrinks to 3 (output image channels)
        self.final_block = nn.Sequential(
            # nn.ReflectionPad2d(3), nn.Conv2d(n_mels, 1, kernel_size=7), nn.Tanh()
            # TODO: without Tanh, for not having output [-1,1]
            nn.ReflectionPad2d(3), nn.Conv2d(n_mels, 1, kernel_size=7)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Feed forward run
        :param input_tensor: The input Tensor
        :param output_size: The output size
        :param random_affine: List of random affine numbers
        :return: The output tensor
        """
        # A condition for having the output at same size as the scaled input is having even output_size

        # Entry block
        feature_map = self.entry_block(input_tensor)

        # Downscale block
        feature_map, downscales = self.downscale_block(
            feature_map, return_all_scales=self.skip
        )

        # Bottleneck (res-blocks)
        feature_map = self.bottleneck_block(feature_map)

        # Upscale block
        feature_map, _ = self.upscale_block(
            feature_map, pyramid=downscales, skip=self.skip
        )

        # Final block
        output_tensor = self.final_block(feature_map)

        return output_tensor

    def save_model(self, model_path):
        cuda = True
        state = { 'net': self.state_dict() if cuda else self.state_dict() }

        torch.save(state,  model_path)

class ResnetBlock(nn.Module):
    """ A single Res-Block module """

    def __init__(self, dim: int, use_bias: bool):
        """
        Init
        :param dim: The dimension
        :param use_bias: Flag to use bias or not
        """
        super(ResnetBlock, self).__init__()

        # A res-block without the skip-connection, pad-conv-norm-relu-pad-conv-norm
        self.conv_block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)
            ),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)
            ),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)
            ),
            nn.BatchNorm2d(dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Feed forward run
        :param input_tensor: The input tensor
        :return: The output tensor
        """
        # The skip connection is applied here
        return input_tensor + self.conv_block(input_tensor)

class RescaleBlock(nn.Module):
    """
    Rescale Block class
    """

    def __init__(self, n_layers: int, scale: Optional[float] = 0.5, n_mels: Optional[int] = 64,
                 use_bias: Optional[bool] = True):
        """
        Init
        :param n_layers: The number of layers
        :param scale: Scale factor
        :param n_mels: Base number of channels
        :param use_bias: Flag to use bias or not
        """
        super(RescaleBlock, self).__init__()

        self.scale = scale

        self.conv_layers = [None] * n_layers

        in_channel_power = scale > 1
        out_channel_power = scale < 1
        i_range = range(n_layers) if scale < 1 else range(n_layers - 1, -1, -1)

        for i in i_range:
            self.conv_layers[i] = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=n_mels * 2 ** (i + in_channel_power),
                        out_channels=n_mels * 2 ** (i + out_channel_power),
                        kernel_size=3,
                        stride=1,
                        bias=use_bias,
                    )
                ),
                nn.BatchNorm2d(n_mels * 2 ** (i + out_channel_power)),
                nn.LeakyReLU(0.2, True))

            self.add_module("conv_%d" % i, self.conv_layers[i])

        if scale > 1:
            self.conv_layers = self.conv_layers[::-1]

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, input_tensor: torch.Tensor,
                pyramid: Optional[torch.Tensor] = None,
                return_all_scales: Optional[bool] = False,
                skip: Optional[bool] = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        :param input_tensor: The input tensor
        :param pyramid: The pyramid tensor
        :param return_all_scales: Flag to return all scales
        :param skip: Flag to skip or not
        :return: Tuple with feature maps and all scales (if return_all_scales is True)
        """
        feature_map = input_tensor
        all_scales = []
        if return_all_scales:
            all_scales.append(feature_map)

        for i, conv_layer in enumerate(self.conv_layers):

            if self.scale > 1.0:
                feature_map = f.interpolate(
                    feature_map, scale_factor=self.scale, mode="nearest"
                )

            feature_map = conv_layer(feature_map)

            if skip:
                feature_map = feature_map + pyramid[-i - 2]

            if self.scale < 1.0:
                feature_map = self.max_pool(feature_map)

            if return_all_scales:
                all_scales.append(feature_map)

        return (feature_map, all_scales) if return_all_scales else (feature_map, None)

