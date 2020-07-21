# arena2d-sim
arena2d-sim is a simple yet efficient 2D simulator build on C++, with configurable agents, highly randomization of levels, sensors and handling of python-based DRL agents.


## Updates
* 01/06/2020:** including asynchronous training. It is now possible to run multiple training instances in parallel. See ... for template
* 01/07/2020:** including A3C agent, based on [Lapan et al. (2018)][lapan]
...

# Instructions

## Installation
- Please follow the instruction from the main arena2d README file for installation of arena2d

## Level Generation
how to create new level, which parameters to change in settings.st when loading level, how to randomize levels

#### SVG levels
there is the possiblity to include svg based maps. show example svg and how its loaded. link to the example and show some tiny example images like office bureau etc.


## Sensor Data
- explain how lidar scan data is constructed, where it can be retrieved by agent. API call or similar, --evtl abstraktion vornehmen
- explanation about how to set noise

## Semantics
You can incorporate classes into the arena for semantic based navigation e.g. human, corridor, door, hallway, etc. 
- explain how one could include different classes (human wanderer, robot wanderer, dynamic door, etc..) and example of how to utilize these semantic information to be considered for training e.g. get closest distance to humna ,etc. .. and where to change it in the reward and penalties.



[lapan]: https://books.google.de/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=lapan+reinforcement+learning&ots=wTgggiYhaD&sig=VjRRQF20if5gCTVjFiuLkw_5mbk#v=onepage&q=lapan%20reinforcement%20learning&f=false
