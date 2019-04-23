# Custom-Convolutions

Convolution is great, but have you tried these?</br>

Implementation of modern variants of convolutional layers in PyTorch</br>

| Method                  | Reference |
| ----------------------- | --------- |
| Adaptively Connected NN |  [Paper](https://arxiv.org/pdf/1904.03579.pdf)|

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
