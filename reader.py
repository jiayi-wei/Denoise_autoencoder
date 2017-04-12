import numpy as np
import os
import time

root_path = '/home/sedlight/workspace/shl/img_data/band'

def load_data(band, patch):
	cmb_path = root_path + band + '_cmb_block_' + patch
	data_list = os.listdir(cmb_path)

	all_files = []

	for filename in data_list:
		cmb_data = os.path.join(cmb_path, filename)
		noise_data = cmb_data.replace('cmb', 'noise')
		all_data = cmb_data.replace('cmb', 'all')
		bunch_data = []
		if os.path.isfile(cmb_data) and os.path.isfile(all_data) and os.path.isfile(noise_data):
			bunch_data.append(noise_data)
			bunch_data.append(all_data)
			bunch_data.append(cmb_data)
		all_files.append(bunch_data)
	
	return all_files


def get_data(pathes):
	all_batch = []
	noise_batch = []
	cmb_batch = []

	for bunch_files in pathes:
		for filename in bunch_files:
			img = np.load(filename)
			img = img[:, :, np.newaxis]
			
			if 'all' in filename:
				all_batch.append(img)
			elif 'noise' in filename:
				noise_batch.append(img)
			elif 'cmb' in filename:
				cmb_batch.append(img)

	all_batch = np.stack(all_batch, axis=0)
	noise_batch = np.stack(noise_batch, axis=0)
	cmb_batch = np.stack(cmb_batch, axis=0)

	return all_batch, noise_batch, cmb_batch
	