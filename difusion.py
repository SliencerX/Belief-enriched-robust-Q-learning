from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2),
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 84,
    # channels = 1,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    './pong_pic',
    train_batch_size = 16,
    train_lr = 1e-4,
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
