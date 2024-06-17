from denoising_diffusion_pytorch.denoising_diffusion_pytorch_VMCond import Unet, GaussianDiffusion, Trainer, num_to_groups
from encoders.clip_model import ModifiedResNet
import torch
import random
import numpy as np
import transforms as Tr
import os
from PIL import Image
from torchvision import transforms as T
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    setup_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_cond_model = ModifiedResNet(layers=[1,2,2,2],output_dim=77,heads=4,input_resolution=64, width=24)

    cond_path = "./gen_cond/"
    base_path = "./generation/"

    #read the first condition velocity map as t20
    file_num = []
    for file in os.listdir(cond_path):
        if os.path.isfile(os.path.join(cond_path,file)):
            if int(file[-6:-4]) == 20:
                file_num.append(file[9:13])
    file_num.sort()

    
    

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

    model_file = 1 # model plan to load
    
    trainer.load(model_file)
    diffusion.image_cond_model = torch.load(f'{trainer.results_folder}/image_cond_model-{model_file}.pth')
    diffusion.image_cond_model = diffusion.image_cond_model.eval()

    def custom_to_tensor(image):
        image = np.expand_dims(image, axis=0)
        tensor = torch.from_numpy(image)

        return tensor

    transform = T.Compose([
        T.Resize((diffusion.image_size, diffusion.image_size)),
        T.Lambda(custom_to_tensor),
        Tr.MinMaxNormalize(trainer.label_min, trainer.label_max)
    ])

    number_of_each_group = 1
    batch_size = 1
    for num in range(number_of_each_group):
        number_of_images_each_group = 18
        number_of_groups = len(file_num)
        for idx in range(number_of_groups): #number of groups
            
            save_path = str(cond_path) + f'/label_sim{file_num[idx]}_t20.npz'
            print(f"Generating the {idx+1} groups of images.")
            for i in range(number_of_images_each_group):
                print("=" * 50)
                batches = [1]
                
                if i  == 0:
                    path_str = save_path
                    velocity = np.load(path_str)
                    array_keys = velocity.files
                    cond_img = velocity[array_keys[0]]
                    cond_img = np.flip(cond_img, axis=0) #deal with the original data
                    cond_img = np.rot90(cond_img, k=3) #deal with the original data
                    cond_img = Image.fromarray(cond_img)
                    cond_img = transform(cond_img)
                    cond_img = cond_img.to(device)
                    cond_img = cond_img.unsqueeze(1)
                else:
                    path_str = save_path
                    velocity = np.load(path_str)
                    array_keys = velocity.files
                    cond_img = velocity[array_keys[0]]
                    cond_img = Image.fromarray(cond_img)
                    cond_img = transform(cond_img)
                    cond_img = cond_img.to(device)
                    cond_img = cond_img.unsqueeze(1)
                    

                all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n, cond_img = cond_img), batches))

                all_images = torch.cat(all_images_list, dim=0)

                all_images = Tr.minmax_denormalize(all_images, trainer.label_min, trainer.label_max)
                all_images = all_images.cpu().numpy()
                all_images = all_images.reshape((diffusion.image_size, diffusion.image_size))
                all_images = cv2.resize(all_images, (401, 141))
                save_path = str(base_path) + f'/sample_g{file_num[idx]}_t{i+3}0_m{model_file}_try{num}.npz'
                np.savez(save_path, label=all_images)



