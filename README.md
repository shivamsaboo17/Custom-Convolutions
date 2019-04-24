# Custom-Convolutions

Convolution is great, but have you tried these?</br>

Implementation of modern variants of convolutional layers in PyTorch</br>

| No. | Method                  | Reference |
| --- | ----------------------- | --------- |
| 1.  | Adaptively Connected NN |  [Paper](https://arxiv.org/pdf/1904.03579.pdf)|
| 2.  | Octave Convolution      |  [Paper](https://arxiv.org/pdf/1904.05049.pdf)|

## Usage
### Adaptively connected layer
```python
from adaptive_conv import AdaptiveConv2D, AdaptiveResidualLayer
# Either provide activation size tuple or use lite version or pass pixel_aware=True
# For the residual layer, by default ReLU and BatchNorm2D are added
adaptive_res_layer = AdaptiveResidualLayer(num_channels, size=None, lite=False, pixel_aware=True)
adaptive_conv_layer = AdaptiveConv2D(input_channels, output_channels,
                                    size=None, lite=True, pixel_aware=False)
```
### Octave convolution
```python
from octave_conv import OctaveConv2D, OctaveResidualLayer
# Either set first as True or last as True or both as False
# If layer is not first it takes [high_tensor, low_tensor] as input else just [high_tensor]
# The layers return [high_tensor, low_tensor] after conv operations
# For the residual layer, by default ReLU and BatchNorm2D are added
first_oct_conv = OctaveConv2D(input_channels, output_channels,
                              alpha_in=0.5, alpha_out=0.5, first=True, last=False)
oct_res_layer = OctaveResidualLayer(num_channels, kernel_size, alpha=0.75)
```
