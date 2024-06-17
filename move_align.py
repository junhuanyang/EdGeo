import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import os
import random

def process(path, dist_path, save_path = './saved_move_align', vis_path = './vis_dir'):
    basefiles = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.endswith('npz'):
            parts = file.split('.')[0].split('_')
            print(parts)
            if len(parts) >3:
                group = parts[1][1:]
                tried = parts[-1][-1]
                if (group, tried) not in basefiles:
                    basefiles.append((group, tried))
    
    for (group, tried) in basefiles:
        print((group, tried))
        move_and_align(dist_path, path, group=group, model=1, tried=tried, save_path = save_path, vis_path = vis_path)



def move_and_align(dist_path, path, thr_s=50, thr_m = 50, group = 1, model = 1, tried = 0, save_path = './saved_move_align', vis_path = './vis_dir'):
    cm_path = 'rainbow256.npy'
    rainbow_cmap = ListedColormap(np.load(cm_path))

    #build base velocity map
    vp1 = np.empty((141, 401), dtype=np.float32)
    vp1[:52, :]     = 2254.5427
    vp1[52:120, :]  = 2463.3452
    vp1[120:141, :] = 2545.3516

    maps = []
    names = []


    # dist_path = '/content/drive/MyDrive/1.SIAM/1.ori_vel/'
    velocity = np.load(dist_path)
    array_keys = velocity.files
    vis = velocity[array_keys[0]]

    #velocity map size in original dataset: [401, 141], change to [141, 401]
    vis = np.flip(vis, axis=0) 
    vis = np.rot90(vis, k=3)

    vis_base = vp1 - vis
    maps.append(vis_base)


    for i in range(2, 20):
        name = f'sample_g{group}_t{i+1}0_m{model}_try{tried}.npz'
        names.append(name)
        if os.path.exists(os.path.join(path, name)):
            velocity = np.load(os.path.join(path, name))
            array_keys = velocity.files
            vis = velocity[array_keys[0]]
        else:
            vis = vp1

        #remove the baseline velocity map
        vis = vp1 - vis
        vis[vis < 0.1] = 0
        maps.append(vis)

    # new plot
    x, y = 5, 4
    fig, ax = plt.subplots(x, y, figsize=(12,12),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.3})


    coordinates = {}
    subfigs = []
    base_bound = []

    for i in range(len(maps)):
        vis3 = maps[i]

        non_zero_pixels = []

        #make the threshold as (max value in the whole group of velocity map) // 3
        thr = np.max(maps) // 3

        #adjust the threshold to avoid the bounding box is too small
        while len(non_zero_pixels) < 500 and thr > 40:
            thr -= 10
            non_zero_pixels = np.argwhere(vis3 > thr)

        #get the bounding box
        if len(non_zero_pixels) > 0:
            min_x, min_y = np.min(non_zero_pixels, axis=0)
            max_x, max_y = np.max(non_zero_pixels, axis=0)

            #adjust the bounding box
            push = 10
            if min_x > push:
                min_x = min_x - push
            if max_x < 140-push:
                max_x += push
            if min_y > push:
                min_y -= push
            if max_y < 400-push:
                max_y += push

            #get the coordinates of the bounding box
            coordinates[i] = [min_x, min_y, max_x, max_y]

            if i == 0:
                bounded_shallow = maps[i][min_x:52, min_y:max_y]
                bounded_middle = maps[i][52:max_x, min_y:max_y]

                base_bound.append(bounded_shallow)
                base_bound.append(bounded_middle)
                img = ax[i // y, i % y].imshow(bounded_shallow, cmap=rainbow_cmap, vmax=1000, vmin=0)
                img = ax[(i+1) // y, (i+1) % y].imshow(bounded_middle, cmap=rainbow_cmap, vmax=1000, vmin=0)
                bounded = maps[i][min_x:max_x, min_y:max_y]
                subfigs.append(bounded)

            elif i == len(maps) - 1:
                for j in range(1, len(maps)):
                    bounded = maps[j][min_x:max_x, min_y:max_y]
                    img = ax[(j+1) // y, (j+1) % y].imshow(bounded, cmap=rainbow_cmap, vmax=1000, vmin=0)
                    subfigs.append(bounded)

    # plt.show()
    plt.savefig(os.path.join(vis_path, f'Crop_g{group}_tried{tried}.png'))
    
    #new plot for the final velocity maps
    x, y = 6, 3
    fig, ax = plt.subplots(x, y, figsize=(16,16),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.3})


    #use the shallow leakage
    base_r_shallow = base_bound[0]

    #filter the shallow leakage
    mask_B_shallow = base_r_shallow > thr_s
    masked_base_r_shallow = base_r_shallow[mask_B_shallow]
    masked_base_r_shallow.sort()

    #filter failed
    if len(masked_base_r_shallow) <= 1:
        raise ValueError(f'{group, tried}')

    # try to compute the CDF (Cumulative Distribution Function) of the filtered shallow leakage values
    try:
        cdf_base_shallow = np.arange(len(masked_base_r_shallow)) / (len(masked_base_r_shallow) - 1)
    except RuntimeWarning as rw:
        raise RuntimeError("RuntimeWarning occurred: " + str(rw))

    #use the middle leakage
    base_r_middle = base_bound[1]

    #filter the middle leakage
    mask_B_middle = base_r_middle > thr_m
    masked_base_r_middle = base_r_middle[mask_B_middle]
    masked_base_r_middle.sort()

    # try to compute the CDF  of the filtered middle leakage values
    try:
        cdf_base_middle = np.arange(len(masked_base_r_middle)) / (len(masked_base_r_middle) - 1)
    except RuntimeWarning as rw:
        print("cdf_base_middle", cdf_base_middle)
        raise RuntimeError("RuntimeWarning occurred: " + str(rw) + f'2 {group, tried}, t{j}')


    # randomly determine a split line
    split = random.randint(subfigs[-1].shape[0] // 3, subfigs[0].shape[0])

    # adjust the split line
    while max_x - (min_x + split - 52) > 120:
        split += 10
    while 52 - split < 0:
        split -= 10
    split_cp = split
    for j in range(1, len(subfigs)):
        # reset the split point for each subfigure

        split = split_cp
        A = subfigs[j].copy()
        min_x, min_y, max_x, max_y = coordinates[len(subfigs) - 1]

        # random split A into 2 parts, 1 for shallow, 1 for middle
        # ensure the split line does not exceed the bounds of the subfigure
        bound = A.shape[0]
        if bound < split:
            split = bound

        # Split the subfigure into shallow and middle parts
        A_shallow = A[: split, :]
        A_middle = A[split:, :]

        # apply threshold masks to the shallow and middle parts
        mask_A_shallow = A_shallow > thr_s
        mask_A_middle = A_middle > thr_m

        # get unique values and their indices for shallow and middle parts
        A_shallow_masked, idx_sort_shallow = np.unique(A_shallow[mask_A_shallow], return_inverse=True)
        A_middle_masked, idx_sort_middle = np.unique(A_middle[mask_A_middle], return_inverse=True)

        # compute the CDF for the masked shallow part
        try:
            cdf_A_shallow = np.arange(len(A_shallow_masked)) / (len(A_shallow_masked) - 1)
        except RuntimeWarning as rw:
            print("cdf_A_shallow",cdf_A_shallow)
            break


        # align the shallow part
        try:
            new_A_s_values = np.interp(cdf_A_shallow, cdf_base_shallow, masked_base_r_shallow)
        except ValueError as ve:
            print("1" * 50)
            print(cdf_A_shallow)
            print(cdf_base_shallow)
            print(masked_base_r_shallow)
            break
        new_A_s = new_A_s_values[idx_sort_shallow]
        new_A_s = np.reshape(new_A_s, newshape=A_shallow[mask_A_shallow].shape)
        A_shallow[mask_A_shallow] = new_A_s


        # compute the CDF for the masked middle part
        try:
            cdf_A_middle = np.arange(len(A_middle_masked)) / (len(A_middle_masked) - 1)
        except RuntimeWarning as rw:
            print("cdf_A_middle", cdf_A_middle)
            break


        # align the middle part
        new_A_m_values = np.interp(cdf_A_middle, cdf_base_middle, masked_base_r_middle)
        new_A_m = new_A_m_values[idx_sort_middle]
        new_A_m = np.reshape(new_A_m, newshape=A_middle[mask_A_middle].shape)
        A_middle[mask_A_middle] = new_A_m


        vis_tmp = np.zeros((141,401))
        # move the leakage
        if min_x - ( min_x + split - 52) >= 0 and min_x + split - 52 > 0:
            move = min_x + split - 52
            vis_tmp[min_x - move: min_x + split-move, min_y:max_y] = A_shallow
            vis_tmp[min_x + split-move: max_x - move, min_y:max_y] = A_middle
        else:
            vis_tmp[min_x: min_x + split, min_y:max_y] = A_shallow
            vis_tmp[min_x + split: max_x, min_y:max_y] = A_middle


        # recover the velocity mapp
        vis_tmp = vp1 - vis_tmp

        #vis
        img = ax[(j-1) // y, (j-1) % y].imshow(vis_tmp, cmap=rainbow_cmap, vmax=2545.3818, vmin=1467.387)
        np.savez(os.path.join(save_path, "revised_"+names[j-1]) , label=vis_tmp)



    cbar_ax = fig.add_axes([0.95, 0.25, 0.02, 0.5])
    clb = plt.colorbar(img, cax=cbar_ax, orientation='vertical')

    clb.ax.set_title('km/s', fontsize=8)
    # plt.show()
    plt.savefig(os.path.join(vis_path, f'result_g{group}_tried{tried}.png'))


#test
if __name__ == '__main__':
    path = './generation'
    dist_path = './label_folder/label_sim0023_t100.npz'
    process(path, dist_path)