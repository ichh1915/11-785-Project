# 11-785-Project
## Dataset
Urban 100 from  "Single Image Super-Resolution from Transformed Self-Exemplars" 

PDF: https://ieeexplore.ieee.org/document/7299156

## Baseline Model 0: SRCNN
"Image Super-Resolution Using Deep Convolutional Networks" 

PDF: https://ieeexplore.ieee.org/abstract/document/7115171/


## Model 1: FSRCNN
"Accelerating the super-resolution convolutional neuralnetwork" 

PDF: http://arxiv.org/abs/1608.00367


## Model 2: SRResNet
"Photo-realistic single image super-resolution using a generative adversarial network"

PDF: http://arxiv.org/abs/1609.04802

## Model 3: CAR-variant
Inspired by

"Learned image downscaling for upscaling using content adaptive resampler"

PDF: https://arxiv.org/pdf/1907.12904.pdf



## RUN
SRCNN
`python run.py --bicubic 1`

FSRCNN
`python run.py --model FSRCNN --bicubic 1`
`python run.py --model FSRCNN --bicubic 0`

SRResNet
`python run.py --model SRResNet --bicubic 1`
`python run.py --model SRResNet --bicubic 0`

CAR-variant
`python run.py --model car --bicubic 1`
`python run.py --model car --bicubic 0`

where bicubic = 1 indicates using the interpolated dataset and bicubic = 0 indicates using the original Urban100 dataset.