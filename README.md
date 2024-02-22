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
