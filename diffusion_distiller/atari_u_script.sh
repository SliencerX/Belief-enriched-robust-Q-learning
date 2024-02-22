# # Train the base model with 1024 diffusion steps

python3 ./train.py --module atari_u --name atari --dname original --batch_size 16 --num_workers 4 --num_iters 200000


# # Model distillation

# ## Distillate to 512 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_0 --base_checkpoint ./checkpoints/atari/original/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 256 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_1 --base_checkpoint ./checkpoints/atari/base_0/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 128 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_2 --base_checkpoint ./checkpoints/atari/base_1/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 64 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_3 --base_checkpoint ./checkpoints/atari/base_2/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 32 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_4 --base_checkpoint ./checkpoints/atari/base_3/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5

# ## Distillate to 16 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_5 --base_checkpoint ./checkpoints/atari/base_4/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5

# ## Distillate to 8 steps
python3 ./distillate.py --module atari_u --diffusion GaussianDiffusionDefault --name atari --dname base_6 --base_checkpoint ./checkpoints/atari/base_5/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5
# # Image generation
python3 ./sample.py --out_file ./images/atari_u_4.png --module atari_u --checkpoint ./checkpoints/atari/base_4/checkpoint.pt --batch_size 1 --clipping_value 1.0
