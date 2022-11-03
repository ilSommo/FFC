import warnings
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple
from .ffc import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor


__all__ = ["FFCInception3", "FFCInceptionOutputs", "FFC_InceptionOutputs", "ffc_inception_v3"]


FFCInceptionOutputs = namedtuple("FFCInceptionOutputs", ["logits", "aux_logits"])
FFCInceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
FFC_InceptionOutputs = FFCInceptionOutputs


class FFCInception3(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
        ratio: float = 0.25,
        lfu: bool = False,
        bn: bool = False,
        relu: bool = False,
        use_se: bool = False
    ) -> None:
        super(FFCInception3, self).__init__()
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of ffc_inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        lfu = lfu
        bn = bn
        relu = relu
        self.ratio = ratio

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = Inception0(192, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, pool_features=32)
        self.Mixed_5c = InceptionA(256, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, pool_features=64)
        self.Mixed_5d = InceptionA(288, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, pool_features=64)
        self.Mixed_6a = InceptionB(288, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, )
        self.Mixed_6b = InceptionC(768, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, )
        self.Mixed_7a = InceptionD(768, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, )
        self.Mixed_7b = InceptionE(1280, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, )
        self.Mixed_7c = InceptionF(2048, ratio_gin=ratio, ratio_gout=ratio, lfu=lfu, bn =bn, relu =relu, )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> FFCInceptionOutputs:
        if self.training and self.aux_logits:
            return FFCInceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> FFCInceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted FFCInception3 always returns FFCInception3 Tuple")
            return FFCInceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class Inception0(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, ratio_gin, ratio_gout, lfu, bn, relu,  
    ) -> None:
        super().__init__()
        self.branch1x1 = FFC_BN(in_channels, 64, ratio_gin=0, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

        self.branch5x5_1 = FFC_BN(in_channels, 48, ratio_gin=0, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch5x5_2 = FFC_BN(48, 64, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = FFC_BN(in_channels, 64, ratio_gin=0, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3dbl_2 = FFC_BN(64, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = FFC_BN(96, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)

        self.branch_pool = FFC_BN(in_channels, pool_features, ratio_gin=0, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = (F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        branch_pool = self.branch_pool(branch_pool)

        outputs_0 = [branch1x1[0], branch5x5[0], branch3x3dbl[0], branch_pool[0]]
        outputs_1 = [branch1x1[1], branch5x5[1], branch3x3dbl[1], branch_pool[1]]
        return (outputs_0, outputs_1)

    def forward(self, x: Tensor) -> Tensor:
        outputs_0, outputs_1 = self._forward(x)
        return (torch.cat(outputs_0, 1), torch.cat(outputs_1, 1))



class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, ratio_gin, ratio_gout, lfu, bn, relu,  
    ) -> None:
        super().__init__()
        self.branch1x1 = FFC_BN(in_channels, 64, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

        self.branch5x5_1 = FFC_BN(in_channels, 48, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch5x5_2 = FFC_BN(48, 64, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = FFC_BN(in_channels, 64, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3dbl_2 = FFC_BN(64, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = FFC_BN(96, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)

        self.branch_pool = FFC_BN(in_channels, pool_features, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = (F.avg_pool2d(x[0], kernel_size=3, stride=1, padding=1), F.avg_pool2d(x[1], kernel_size=3, stride=1, padding=1))
        branch_pool = self.branch_pool(branch_pool)

        outputs_0 = [branch1x1[0], branch5x5[0], branch3x3dbl[0], branch_pool[0]]
        outputs_1 = [branch1x1[1], branch5x5[1], branch3x3dbl[1], branch_pool[1]]
        return (outputs_0, outputs_1)

    def forward(self, x: Tensor) -> Tensor:
        outputs_0, outputs_1 = self._forward(x)
        return (torch.cat(outputs_0, 1), torch.cat(outputs_1, 1))


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, ratio_gin, ratio_gout, lfu, bn, relu) -> None:
        super().__init__()
        self.branch3x3 = FFC_BN(in_channels, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = FFC_BN(in_channels, 64, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3dbl_2 = FFC_BN(64, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = FFC_BN(96, 96, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = (F.max_pool2d(x[0], kernel_size=3, stride=2), F.max_pool2d(x[1], kernel_size=3, stride=2))

        outputs0 = [branch3x3[0], branch3x3dbl[0], branch_pool[0]]
        outputs1 = [branch3x3[1], branch3x3dbl[1], branch_pool[1]]
        return outputs0, outputs1

    def forward(self, x: Tensor) -> Tensor:
        outputs0, outputs1 = self._forward(x)
        return (torch.cat(outputs0, 1), torch.cat(outputs1, 1))


class InceptionC(nn.Module):
    def __init__(
        self, in_channels: int, channels_7x7: int, ratio_gin, ratio_gout, lfu, bn, relu
        ) -> None:
        super().__init__()
        self.branch1x1 = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = FFC_BN(in_channels, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch7x7_2 = FFC_BN(c7, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = FFC_BN(c7, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = FFC_BN(in_channels, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch7x7dbl_2 = FFC_BN(c7, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = FFC_BN(c7, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = FFC_BN(c7, c7, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = FFC_BN(c7, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = (F.avg_pool2d(x[0], kernel_size=3, stride=1, padding=1), F.avg_pool2d(x[1], kernel_size=3, stride=1, padding=1))
        branch_pool = self.branch_pool(branch_pool)

        outputs0 = [branch1x1[0], branch7x7[0], branch7x7dbl[0], branch_pool[0]]
        outputs1 = [branch1x1[1], branch7x7[1], branch7x7dbl[1], branch_pool[1]]
        return outputs0, outputs1

    def forward(self, x: Tensor) -> Tensor:
        outputs0, outputs1 = self._forward(x)
        return (torch.cat(outputs0, 1), torch.cat(outputs1, 1))


class InceptionD(nn.Module):
    def __init__(self, in_channels: int, ratio_gin, ratio_gout, lfu, bn, relu) -> None:
        super().__init__()
        self.branch3x3_1 = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3_2 = FFC_BN(192, 320, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, stride=2)

        self.branch7x7x3_1 = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch7x7x3_2 = FFC_BN(192, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = FFC_BN(192, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = FFC_BN(192, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = (F.max_pool2d(x[0], kernel_size=3, stride=2), F.max_pool2d(x[1], kernel_size=3, stride=2))
        outputs0 = [branch3x3[0], branch7x7x3[0], branch_pool[0]]
        outputs1 = [branch3x3[1], branch7x7x3[1], branch_pool[1]]
        return outputs0, outputs1

    def forward(self, x: Tensor) -> Tensor:
        outputs0, outputs1 = self._forward(x)
        return (torch.cat(outputs0, 1), torch.cat(outputs1, 1))


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, ratio_gin, ratio_gout, lfu, bn, relu) -> None:
        super().__init__()
        self.branch1x1 = FFC_BN(in_channels, 320, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

        self.branch3x3_1 = FFC_BN(in_channels, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3_2a = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = FFC_BN(in_channels, 448, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3dbl_2 = FFC_BN(448, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = (torch.cat([branch3x3[0][0], branch3x3[1][0]], 1), torch.cat([branch3x3[0][1], branch3x3[1][1]], 1))

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = (torch.cat([branch3x3dbl[0][0], branch3x3dbl[1][0]], 1), torch.cat([branch3x3dbl[0][1], branch3x3dbl[1][1]], 1))

        branch_pool = (F.avg_pool2d(x[0], kernel_size=3, stride=1, padding=1), F.avg_pool2d(x[1], kernel_size=3, stride=1, padding=1))
        branch_pool = self.branch_pool(branch_pool)

        outputs0 = [branch1x1[0], branch3x3[0], branch3x3dbl[0], branch_pool[0]]
        outputs1 = [branch1x1[1], branch3x3[1], branch3x3dbl[1], branch_pool[1]]
        return outputs0, outputs1

    def forward(self, x: Tensor) -> Tensor:
        outputs0, outputs1 = self._forward(x)
        return (torch.cat(outputs0, 1), torch.cat(outputs1, 1))


class InceptionF(nn.Module):
    def __init__(self, in_channels: int, ratio_gin, ratio_gout, lfu, bn, relu) -> None:
        super().__init__()
        self.branch1x1 = FFC_BN(in_channels, 320, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

        self.branch3x3_1 = FFC_BN(in_channels, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3_2a = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = FFC_BN(in_channels, 448, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.branch3x3dbl_2 = FFC_BN(448, 384, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = FFC_BN(384, 384, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = FFC_BN(in_channels, 192, ratio_gin=ratio_gin, ratio_gout=0, lfu=lfu, bn =bn, relu =relu, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3)[0],
            self.branch3x3_2b(branch3x3)[0],
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl)[0],
            self.branch3x3dbl_3b(branch3x3dbl)[0],
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = (F.avg_pool2d(x[0], kernel_size=3, stride=1, padding=1), F.avg_pool2d(x[1], kernel_size=3, stride=1, padding=1))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1[0], branch3x3, branch3x3dbl, branch_pool[0]]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, ratio_gin, ratio_gout, lfu, bn, relu, ffc_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if ffc_block is None:
            ffc_block = FFC_BN
        self.conv0 = ffc_block(in_channels, 128, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu, bn =bn, relu =relu, kernel_size=1)
        self.conv1 = ffc_block(128, 768, ratio_gout=0, ratio_gin=ratio_gin, lfu=lfu, bn =bn, relu =relu, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = (F.avg_pool2d(x[0], kernel_size=5, stride=3), F.avg_pool2d(x[1], kernel_size=5, stride=3))
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x[0], (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



def ffc_inception_v3(*, weights = None, progress: bool = True, **kwargs: Any) -> FFCInception3:
    """
    Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the ffc_inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        weights (:class:`~torchvision.models.FFCInception_V3_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.FFCInception_V3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.FFCInception3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.FFCInception_V3_Weights
        :members:
    """

    original_aux_logits = kwargs.get("aux_logits", True)

    model = FFCInception3(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model