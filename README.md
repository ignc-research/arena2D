# arena2D
A plattform to ease DRL research.

Arena2D is a research framework for fast development and training of reinforcement learning algorithms for autonomous navigation. 
It aims to ease the overall pipeline and make training more efficient using simple 2D environments together with optimization modules, 
which users can freely experiment with. We have incorporated novel ideas from state of the art research and arena2d provides APIs, which every user can 
use to integrate new ideas.

Our design principles are:

- Efficient simulation: Training in arena2D require less time compared to 3D simulations while achieving similar results on lidar scan data.
- Flexible development: Make it easy for new users to try out research ideas.
- Easy experimentation: Make it easy for new users to test with benchmarks and run richful evaluations
- Modular expandability: New functionalities can be build in easily, therefore we provide API.




## Table of contents
   1. [Motivation](#Updates)
   1. [Citing Arena2D](#citing-arena)
   1. [Installation](#installation)
   1. [Docker Setup](#docker-setup)
   1. [Training and Testing Agent](#example)
   1. [Documentation](#documentation)
   1. [Details](#details)
   1. [Data](#data)
   1. [Baselines](#baselines)
   1. [License](#license)
   1. [Acknowledgments](#acknowledgments)
   1. [References](#references-and-citation)


## Citing Arena2D
If you use the Habitat platform in your research, please cite the following [paper](https://arxiv.org/abs/ TODO. paper auf arxiv hochladen!!):

```
@inproceedings{arena2d,
  title     =     {Arena2D: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    =     {},
  booktitle =     {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =     {2020}
}
```

## Updates
* 01/06/2020:** including asynchronous training. It is now possible to run multiple training instances in parallel. See ... for template
* 01/07/2020:** including A3C agent, based on [Lapan et al. (2018)][lapan]
...

# Instructions

## Installation
- Create conda environment
- Install neccessary dependencies
- clone stable version from github
...
```
conda create -new ....
```

#### Ubuntu 

```
sudo apt-get update && sudo apt-get install cmake zlib1g-dev
pip install absl-py atari-py gin-config gym opencv-python tensorflow==1.15
```

#### Mac OS X

```
brew install cmake zlib
pip install absl-py atari-py gin-config gym opencv-python tensorflow==1.15
```

## Training and Testing Agents
For training and testing preexisting or new agents please see Running.md

## Documentation
Overall workflow of arena2D

<p align="center">
  <img src='img/arena2d.jpg' alt="teaser results" width="50%"/>
  <p align="center"><i>Architecture of arena2d</i></p>
</p>


[lapan]: https://books.google.de/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=lapan+reinforcement+learning&ots=wTgggiYhaD&sig=VjRRQF20if5gCTVjFiuLkw_5mbk#v=onepage&q=lapan%20reinforcement%20learning&f=false
