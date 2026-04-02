# ResNet18 Top 5 Layers by MAC Count

MACs calculated as: `params × output_height × output_width` for Conv2d layers.

| Layer Name | MACs | Parameters |
|------------|------|------------|
| Conv2d: 1-1 | 118,013,952 | 9,408 |
| Conv2d: 3-1 | 115,605,504 | 36,864 |
| Conv2d: 3-4 | 115,605,504 | 36,864 |
| Conv2d: 3-7 | 115,605,504 | 36,864 |
| Conv2d: 3-10 | 115,605,504 | 36,864 |

**Note:** 12 Conv2d layers are tied at 115,605,504 MACs. This includes layers from all four residual stages (3-1 through 3-49), as the architecture balances reduced spatial dimensions with increased channel counts. The first conv (1-1) has the highest MACs due to its large 112×112 output despite having fewer parameters.

## Arithmetic Intensity: Conv2d 1-1

```
FLOPs  = 2 × MACs = 2 × 118,013,952 = 236,027,904

Bytes  = (input + weights + output) × 4 bytes/element
       = (3×224×224 + 9,408 + 64×112×112) × 4
       = (150,528 + 9,408 + 802,816) × 4
       = 962,752 × 4
       = 3,851,008

AI = FLOPs / Bytes = 236,027,904 / 3,851,008 = 61.29 FLOPs/byte
```
