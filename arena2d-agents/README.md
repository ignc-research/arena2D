# arena2D-agent




## Updates
* 01/06/2020:** including asynchronous training. It is now possible to run multiple training instances in parallel. See ... for template
* 01/07/2020:** including A3C agent, based on [Lapan et al. (2018)][lapan]
...

# Instructions

## Installation
- follow instructions at main page

## New Agent
- example how to build new agent, pre step post step, example, how to retrieve sensor data, where to define reward, penalties, hyperparameters.

- Actions: how to change actions example code snippet with discrete and continous action

## Asynchronous Agents
- explain asynchronous parameters to set for training, give some example. link to a3c, 



## Network Design
you can explore different network architectures. currently there are FC, CNN, LSTM implemented 
- what is input output of the networks, maybe some abstraction such that user can easily try out different combinations
- what to consider when making new architecture (input, output, interface between network and agent)
- interface between different agents
- example code


## Training Agents
- commands to run training, all additional flags, parameters in command line 
For training and testing preexisting or new agents please have a look at this example file

## Test and Evaluate Agents
- commands to run eval script
- tensorboard command 
- maybe include video_dir to see on tensorboard if working remote etc. 

## Documentation
Overall workflow of creating new agent, e.g. first define new class out of template, then if you want to change network, then change that there, 

<p align="center">
  <img src='res/img/habitat_api_structure.png' alt="teaser results" width="100%"/>
  <p align="center"><i>Architecture of Habitat-API</i></p>
</p>
