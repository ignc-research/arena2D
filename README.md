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


# Updates
01/06/2020:** including asynchronous training. It is now possible to run multiple training instances in parallel. See ... for template
01/07/2020:** including A3C agent, based on [Lapan et al. (2018)][lapan]
...

# Instructions

## Installation
Install neccessary dependencies
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






[lapan]: https://books.google.de/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=lapan+reinforcement+learning&ots=wTgggiYhaD&sig=VjRRQF20if5gCTVjFiuLkw_5mbk#v=onepage&q=lapan%20reinforcement%20learning&f=false
