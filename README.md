# Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations

This is the official code for our paper [Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations](https://openreview.net/forum?id=7gDENzTzw1&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)) published at ICLR 2024. The default environment for Atari is Pong, you need to change the environment by changing the configs and use coresponding diffusion model.

This repo contains PF-RNN code from [Particle Filter Recurrent Neural Networks(Ma et.al., 2019)](https://github.com/Yusufma03/pfrnns) and code from Progressive Distillation for [Fast Sampling of Diffusion Models(Salimans and HO, 2022)](https://github.com/Hramchenko/diffusion_distiller) 

We include pretrained PF-RNN, diffusion models and our models in pretrained folders
## Train PF-RNN Model
```
python3 pfrnns/main.py
```
## Train Diffusion Model
### First Generate Trajectory
```
python3 gen_atari_pic.py --config config/Pong_ours.json
python3 diffuion.py
```

### Train Diffusion Distiller Model
Copy generated atari pics into diffusion_distiller folder
```
bash diffusion_distiller/atari_u_script.sh
```

## Train our model
### BP-DQN
```
python3 train.py --config config/Grid_continous_ours.json
```
### DP-DQN-O
```
python3 train_atari.py --config config/Pong_ours.json
```
### DP-DQN-F
```
python3 train_atari.py --config config/Pong_ours.json
```


## Test our model
### BP-DQN
```
python3 test_gridmaze.py --config config/Grid_continous_ours.json
```
### DP-DQN
```
python3 test_atari.py --config config/Pong_ours.json
```

## To cite our work
```
@inproceedings{
sun2024beliefenriched,
title={Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations},
author={Xiaolin Sun and Zizhan Zheng},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=7gDENzTzw1}
}
```
