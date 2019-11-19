# Avatar-GAN (RoboCoDraw)
This is the open-source repository of the project Avatar-GAN from the paper RoboCoDraw: Robotic Avatar Drawing with GAN-based Style Transfer andTime-efficient Path Optimization published in AAAI 2020.

Please cite our paper if this repository is useful for your project:

TODO: bibtex of paper

# Avatar-GAN system

Avatar-GAN was implemented based on Cycle-GAN proposed in paper [[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee]](https://arxiv.org/pdf/1703.10593.pdf) [[github]](https://github.com/xhujoy/CycleGAN-tensorflow).

## Prerequisites
- tensorflow-gpu 1.13.1
- numpy 1.17.1
- scipy 1.3.1
- Pillow 6.1.0
- scikit-image 0.16.2 

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/Psyhce-mia/Avatar-GAN
cd Avatar-GAN/GAN
```

### Train 
- Download the training dataset [Avatar-GAN dataset](not available yet) to Avatar-GAN/datasets/ folder

- Train a model:
```bash
python main.py
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

### Test
- Finally, test the model:
```bash
python main.py --phase=test 
```

## Datasets
The datasets used in Avatar-GAN were generated from:

- `Chicago Face Dataset (CFD)`: [CFD dataset](https://chicagofaces.org/default/).
- `Avataaars Library`: [Avataaars](https://avataaars.com/).
- `CUHK Face Sketch Database`: [CUFS](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html).

Note that we have applied preprocesses to the raw datasets.


# RoboCoDraw system
