# Avatar-GAN (RoboCoDraw)
This is the open-source repository of the project Avatar-GAN from the paper RoboCoDraw: Robotic Avatar Drawing with GAN-based Style Transfer andTime-efficient Path Optimization published in AAAI 2020.

Please cite our paper if this repository is useful for your project:

TODO: bibtex of paper

# Avatar-GAN system

Avatar-GAN was implemented based on Cycle-GAN proposed in paper [[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee]](https://arxiv.org/pdf/1703.10593.pdf) [[github]](https://github.com/xhujoy/CycleGAN-tensorflow).

## Prerequisites
- tensorflow r1.1
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/Psyhce-mia/Avatar-GAN
cd Avatar-GAN
```

### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./download_dataset.sh horse2zebra
```
- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

### Test
- Finally, test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra --phase=test --which_direction=AtoB
```

## Training and Test Details
To train a model,  
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ 
```
Models are saved to `./checkpoints/` (can be changed by passing `--checkpoint_dir=your_dir`).  

To test the model,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ --phase=test --which_direction=AtoB/BtoA
```

## Datasets
The datasets used in Avatar-GAN are generated from:

- `Chicago Face Dataset (CFD)`: [CFD dataset](https://chicagofaces.org/default/).
- `Avataaars Library`: [Avataaars](https://avataaars.com/).


# RoboCoDraw system
