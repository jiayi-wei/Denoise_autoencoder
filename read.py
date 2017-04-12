import numpy as np
import os

# input image dimensions
input_dim = 1024

def helper(num,end,left,top,right,down):
    if end<=4:
        return top+1-num%2,left+1-num/2
    elif num<end/4:
        return helper(num,end/4,(left+right)/2,(top+down)/2,right,down)
    elif num<end/2:
        return helper(num-end/4,end/4,(left+right)/2,top,right,(top+down)/2)
    elif num<3*end/4:
        return helper(num-end/2,end/4,left,(top+down)/2,(left+right)/2,down)
    else:
        return helper(num-3*end/4,end/4,left,top,(left+right)/2,(top+down)/2)

def reorder(in_data):
    img = np.zeros((input_dim,input_dim),dtype=np.float32)
    for i in range(input_dim*input_dim):
        row,col=helper(i,input_dim*input_dim,0,0,input_dim,input_dim)
        img[row][col]=in_data[i]
    return img


def get_data(pathes):
    all_batch = []
    noise_batch = []
    cmb_batch = []

    for bunch_files in pathes:
        for i ,filename in enumerate(bunch_files):
            in_data = np.fromfile((filename),dtype=np.float32)
            img = reorder(in_data)
            img = img[:, :, np.newaxis]
            if i==0:
                all_batch.append(img)
            elif i==1:
                noise_batch.append(img)
            elif i==2:
                cmb_batch.append(img)
            
    all_batch = np.stack(all_batch, axis=0)
    noise_batch = np.stack(noise_batch, axis=0)
    cmb_batch = np.stack(cmb_batch, axis=0)
    
    return all_batch, noise_batch, cmb_batch


def load_data(band, patch):
    root_path = "/home/sedlight/workspace/shl/fits_data/band" + band
    cmb_path = "band" + band + "_cmb_block_" + patch
    cmb_absolute_path = os.path.join(root_path, cmb_path)
    noise_path = "band" + band + "_noise_block_" + patch
    noise_absolute_path = os.path.join(root_path, noise_path)
    all_path = "band" + band + "_all_block_" + patch
    all_absolute_path = os.path.join(root_path, all_path)

    all_file = []
    file_list = os.listdir(cmb_absolute_path)

    for filename in file_list:
        bunch_files = []

        all_name = filename.replace("cmb", "all")
        bunch_files.append(os.path.join(all_absolute_path, all_name))

        noise_name = filename.replace("cmb", "noise")
        bunch_files.append(os.path.join(noise_absolute_path, noise_name))

        bunch_files.append(os.path.join(cmb_absolute_path, filename))

        all_file.append(bunch_files)
    return all_file

if __name__ == "__main__":
    print load_data('1', '5')
