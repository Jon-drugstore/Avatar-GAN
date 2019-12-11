# Avatar-GAN (RoboCoDraw)
![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)
This is the open-source repository of the project Avatar-GAN from the paper **RoboCoDraw: Robotic Avatar Drawing with GAN-based Style Transfer andTime-efficient Path Optimization** published in AAAI 2020.

Please cite our paper if this repository is useful for your project:

TODO: bibtex of paper

**Note**: The datasets and checkpoints in Avatar-GAN system and RoboCoDraw system are **different versions**. The datasets and checkpoints in Avatar-GAN system (code in */GAN* folder) are the same as that in the paper for reproducibility. The datasets in RoboCoDraw system were further processed (e.g. automatic background removal) for automatic drawing. 

Feel free to contact Tianying (wty00678@gmail.com) or Wei Qi (toh_wei_qi@scei.a-star.edu.sg).

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

## Prerequisites
- tensorflow 1.13.1
- numpy 1.17.1
- opencv 4.1.1
- deap 1.3.0

## Getting Started
### Installation
Clone this repo:
```bash
git clone https://github.com/Psyhce-mia/Avatar-GAN
```
### Running the RoboCoDraw system
To run the RoboCoDraw system with a UR robot, install `python-urx`. Ensure the robot end effector with the marker is at the bottom left corner of the drawing surface before executing the code with the `--robot_drawing` flag:  
```bash
python run_RoboCoDraw.py --filename=CFD/gray1.jpg --opt_algo=rkgaLK --drawing_size=0.25 --table_surface_z=0.0 --robot_drawing 
```
**Note:** 
- `--filename` sets the input image filename
- `--opt_algo` sets the optimization algorithm used for robot drawing (greedy, greedy2opt, greedyLK, rkga2opt, rkgaLK)
- `--drawing_size` sets the size of the output robot drawing (in meters)
- `--table_surface_z` sets the position of drawing table surface along the robot's z-axis


To run the RoboCoDraw system (style transfer + optimization) without executing the final robot drawing, run the following:
```bash
python run_RoboCoDraw.py --filename=CFD/gray1.jpg --opt_algo=rkgaLK
```
