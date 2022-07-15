# Deep Dynamic Transformer Model for Multi-Start Team Orienteering Problem
 Deep Dynamic Transformer Model (DDTM) for Multi-Start Team Orienteering Problem (MSTOP). 
 Prizes are either constant or uniformly distributed. 
  **Training** based on REINFORE algorithm with greedy rollout baseline (A), greedy rollout baseline with maximum entropy objective (B), multiple-sample baseline with replacement (C), instance-augmentation baseline (D), and instance-augmentation baseline with maximum entropy objecitve (E). 

## Usage
---
### Running DDTM
To run DDTM for solving MSTOP instances, first go to `run.py` and uncomment `line 16` as
```
from nets.attention_model import AttentionModel
```
### Running AM
To run vanilla AM for solving TSP/CVRP instances, uncomment `line 17` in `run.py` as
```
from nets.attention_model_original import AttentionModel
```
## Dependencies
---
- Python >= 3.8
- Numpy
- SciPy
- Pytorch = 1.9.0
- tqdm
- tensorboard_logger 
- Matplotlib = 3.4.3

## Acknowledgements
---
This repository contains adaptations of the following repositories as basework
- [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)
- [TSP_Transformer](https://github.com/xbresson/TSP_Transformer)
- [POMO](https://github.com/yd-kwon/POMO)