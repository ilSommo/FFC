# Fast Fourier Convolution (FFC) for Image Classification on ResNet-50 and Inception v3

## To run the code

For ResNet-50
```bash
python main_224.py -a [model_name] [dataset_folder] [other options]
```

For Inception v3
```bash
python main_299.py -a [model_name] [dataset_folder] [other options]
```

## Inception v3
The vanilla version of Inception v3 consists of a combination of:
 * 2D convolution blocks (Conv2d + BatchNorm2d + relu)
 * Max pooling layers
 * Average pooling layers
 * Linear layers
 * Inception blocks (A, B, C, D, E, AUX)

The Inception blocks are in turn composed by a mixture of 2D convolution blocks, average pooling and concatenation.

The standard order of blocks is the following:

```python
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
```

With
```python
self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
self.Mixed_5b = inception_a(192, pool_features=32)
self.Mixed_5c = inception_a(256, pool_features=64)
self.Mixed_5d = inception_a(288, pool_features=64)
self.Mixed_6a = inception_b(288)
self.Mixed_6b = inception_c(768, channels_7x7=128)
self.Mixed_6c = inception_c(768, channels_7x7=160)
self.Mixed_6d = inception_c(768, channels_7x7=160)
self.Mixed_6e = inception_c(768, channels_7x7=192)
self.AuxLogits: Optional[nn.Module] = None
if aux_logits:
    self.AuxLogits = inception_aux(768, num_classes)
self.Mixed_7a = inception_d(768)
self.Mixed_7b = inception_e(1280)
self.Mixed_7c = inception_e(2048)
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.dropout = nn.Dropout(p=dropout)
self.fc = nn.Linear(2048, num_classes)
```

## FFC Inception v3

When applying the FFC intuition to the Inception v3 architecture, the basic idea is to substitute standard 2d convolutions with Fast Fourier Convolutions. Unfortunately, not all convolutions can be converted to FFC: the limitations arises from the fact that the global features need to be summed to the local ones as part of the structure of the block, therefore the input spatial dimensions must be multiples of the output spatial dimensions. Therefore, some of the stemming convolutions can be substituted, while others can not. For the sake of consistency, all stemming convolutions will be left as `BasicConv2d`, while all of the convolutions of the Inception blocks will be converted.

Since the first and final FFCs must be treated separately (the first one has no global input and the last one has no global output) two new Inception block flavors are added: `Inception0` and `InceptionF`. The main difference with their standard counterparts (`InceptionA` and `InceptionE`) is therefore the global channels management, while the rest of the structure will be kept almost identical. Obviously, contrary to all other Inception blocks whose inputs and ouputs are now tuples of local and globals tensors, `Inception0` takes a single tensor as input, while `InceptionF` gives a single tensor as output.

```python
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
```

Some new parameters are now passed to the Inception blocks:
 * `ratio_gin` and `ratio_gout` provide the ratio of global channels in input and output
 * `lfu`, `bn`, and `relu` are flags to activate Local Fourier Unit, Batch Norm and Relu respectively

In general, in order to mimick the structure of standard Inception v3 BasicConc2d blocks, batch norm and relu should be always on: after some testing it was chosen to leave relu always on but to try different combinations of bn and lfu, in order to evaluate their impact.

## Training

The training was done using the standard script provided by PyTorch to evaluate Imagenet performances. Since training on the whole of Imagenet was not feasible, it was chosen to evaluate the models on two subsets of Imagenet: Imagenette (a very simple 10-class dataset) and Imagewoof (a more challengin 10-class dataset).
The tested models were:
* ResNet-50
* FFC-ResNet-50
* FFC-ResNet-50 with LFU
* Inception v3
* FFC-Inception v3
* FFC-Inception v3 with LFU
* FFC-Inception v3 with Batch Norm
* FFC-Inception v3 with Batch Norm and LFU

For ResNet-50 and FFC-ResNet-50 the following parameters were used:
* epochs = 90
* batch size = 128
* learning rate = 0.1
* lr_steps = [30, 60, 80]
* warmup epochs = 5
* weight decay = 1e-4
* optimizer = SGD
* image crop size = 224x224

For Inception v3 and FFC-Inception v3 the following parameters were used:
* epochs = 90
* batch size = 128
* learning rate = 0.01
* lr_steps = [30, 60, 80]
* warmup epochs = 5
* weight decay = 1e-4
* optimizer = RMSprop
* image crop size = 299x299

Regarding regularization: in both cases a random resized crop was used in training, with random horizontal flipping as well as brightness and saturation jitter. For validation, a center cropping with a ratio of 7/8 was preferred, with no other changes applied to the input image.
Regarding the loss: for ResNet-50 standard Cross-Entropy was used on the output, while for Inception the Cross-Entropy was calculated separately on the final and auxiliary output, before being summed.

## Results

To evaluate the models, top-1 and top-3 accuracy were used: top-5 accuracy was not considered since the number of class is only 10.

### ResNet-50 on ImageNette
| Model | Top-1 Acc | Top-3 Acc |
|---|---|---|
| ResNet-50 | 89.070 | 97.121 |
| FFC-ResNet-50 | **89.783** | 97.274 |
| FFC-ResNet-50 + LFU | 88.943 | **97.299** |

### Inception v3 on ImageNette
| Model | Top-1 Acc | Top-3 Acc |
|---|---|---|
| Inception v3 | 86.930 | 96.662 |
| FFC-Inception v3 | 85.503 | 95.771 |
| FFC-Inception v3 + LFU | 85.885 | 96.127 |
| FFC-Inception v3 + BN | **90.268** | **97.631** |
| FFC-Inception v3 + BN + LFU | 88.815 | 97.427 |

### ResNet-50 on ImageWoof
| Model | Top-1 Acc | Top-3 Acc |
|---|---|---|
| ResNet-50 | 80.758 | 95.062 |
| FFC-ResNet-50 | **81.395** | **95.393** |
| FFC-ResNet-50 + LFU | 81.267 | 94.884 |

### Inception v3 on ImageWoof
| Model | Top-1 Acc | Top-3 Acc |
|---|---|---|
| Inception v3 | 75.388 | 92.695 |
| FFC-Inception v3 | 71.519 | 92.314 |
| FFC-Inception v3 + LFU | 65.182 | 89.972 |
| FFC-Inception v3 + BN | **80.402**  | **94.655** |
| FFC-Inception v3 + BN + LFU | 68.796 | 90.099 |