from denoising_diffusion_pytorch.denoising_diffusion_pytorch_VMCond import Unet, GaussianDiffusion, Trainer
from encoders.clip_model import ModifiedResNet
import torch
import random
import numpy as np


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_cond_model = ModifiedResNet(layers=[1,2,2,2],output_dim=77,heads=4,input_resolution=64, width=24)

    setup_seed(0)

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        out_dim= 1,
        use_spatial_transformer = True,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 300,           # number of steps
        sampling_timesteps = 60,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = 'pred_noise',
        image_cond_model = image_cond_model,
        training = True
    )

    trainer = Trainer(
        diffusion,
        folder = './label_folder',
        des_file='./example.csv',
        train_batch_size = 8,
        train_lr = 8e-5,
        train_num_steps = 300000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        results_folder = './results/',
        augment_horizontal_flip = False,
        label_min=1467.387,
        label_max=2545.3818,
        save_and_sample_every = 10000,
    )

    trainer.train()
