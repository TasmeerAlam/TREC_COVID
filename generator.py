'''
from i3d import Inception_Inflated3d



skip_layer_names = ['Mixed_5b','Mixed_4d','Mixed_3b','Conv3d_1a_7x7']
encoder = Inception_Inflated3d(include_top=False,input_shape=(16,224,224,3),weights='rgb_imagenet_and_kinetics')
inp1_shape = encoder.layers[-2].output.shape
inp2_shape = encoder.get_layer(skip_layer_names[0]).output.shape
inp3_shape = encoder.get_layer(skip_layer_names[1]).output.shape
inp4_shape = encoder.get_layer(skip_layer_names[2]).output.shape
inp5_shape = encoder.get_layer(skip_layer_names[3]).output.shape
'''

import os
from glob import glob
import cv2
import numpy as np


def adjustData(img,mask,flag_multi_class,num_class):

    if(flag_multi_class):

        img = img / 255

        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]

        new_mask = np.zeros(mask.shape + (num_class,))

        for i in range(num_class):

            #for one pixel in the image, find the class in mask and convert it into one-hot vector

            #index = np.where(mask == i)

            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)

            #new_mask[index_mask] = 1

            new_mask[mask == i,i] = 1

        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))

        mask = new_mask

    elif(np.max(img) > 1):

        img = img / 255

        mask = mask /255

        mask[mask > 0.5] = 1

        mask[mask <= 0.5] = 0

    return (img,mask)

		


def generator(video_paths,number_of_frames,batch_size):

	
	frame_batch = list()

	mask_frame_batch = list()

	while True:
		for i in range(len(video_paths)):

			cap = cv2.VideoCapture(video_paths[i])

			mask_path = os.path.join('.','mask',os.path.split(video_paths[i])[-1])

			mask_cap = cv2.VideoCapture(mask_path)

			frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



			

			for j in range(0,frame_count,number_of_frames):
				frame_list = list()

				mask_frame_list = list()
				
				if j == frame_count:
					break

				if frame_count - j < number_of_frames:
					for k in range(frame_count-number_of_frames,frame_count):
						cap.set(1,k)

						mask_cap.set(1,k)

						res, frame = cap.read()

						mask_res, mask_frame = mask_cap.read()

						frame = cv2.resize(frame,(224,224))
						mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
						

						mask_frame = cv2.resize(mask_frame,(224,224))

						

						mask_frame = np.expand_dims(mask_frame, axis=-1)

						frame_list.append(frame)

						mask_frame_list.append(mask_frame)

				else:
					for k in range(j,j+number_of_frames):
					
						cap.set(1,k)
						mask_cap.set(1,k)

						res, frame = cap.read()

						mask_res, mask_frame = mask_cap.read()

						#print(frame.shape)
						frame = cv2.resize(frame,(224,224))
						mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
						
						mask_frame = cv2.resize(mask_frame,(224,224))

						mask_frame = np.expand_dims(mask_frame, axis=-1)
						
						frame_list.append(frame//255)

						mask_frame_list.append(mask_frame//255)

				frame_batch.append(np.array(frame_list))

				mask_frame_batch.append(np.array(mask_frame_list))

				if len(frame_batch) == batch_size:
					yield [np.array(frame_batch), np.array(mask_frame_batch)]

					mask_frame_batch = list()

					frame_batch = list()

def generator_with_skip_data(video_paths,encoder,number_of_frames,batch_size):

	
	frame_batch = list()

	mask_frame_batch = list()

	while True:
		for i in range(len(video_paths)):

			cap = cv2.VideoCapture(video_paths[i])

			mask_path = os.path.join('.','mask',os.path.split(video_paths[i])[-1])

			mask_cap = cv2.VideoCapture(mask_path)

			frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



			print(frame_count)

			for j in range(0,frame_count,number_of_frames):
				frame_list = list()

				mask_frame_list = list()
				
				if j == frame_count:
					break

				if frame_count - j < number_of_frames:
					for k in range(frame_count-number_of_frames,frame_count):
						cap.set(1,k)

						mask_cap.set(1,k)

						res, frame = cap.read()

						mask_res, mask_frame = mask_cap.read()

						frame.reshape((224,224,3))
						mask_frame.reshape((224,224,3))

						frame_list.append(frame)

						mask_frame_list.append(mask_frame)

				else:
					for k in range(j,j+number_of_frames):
					
						cap.set(1,k)
						mask_cap.set(1,k)

						res, frame = cap.read()

						mask_res, mask_frame = mask_cap.read()

						frame.reshape((224,224,3))
						mask_frame.reshape((224,224,3))

						frame_list.append(frame)

						mask_frame_list.append(mask_frame//255)

				frame_batch.append(frame_list)

				mask_frame_batch.append(mask_frame_list)

				if len(frame_batch) == batch_size:

					yield frame_batch, mask_frame_batch

					mask_frame_batch = list()

					frame_batch = list()
					

VIDEOS_PATHS = os.path.join('.','processed_data','*')

VIDEOS_PATHS_LIST = [videos for videos in glob(VIDEOS_PATHS)]



#c = generator(VIDEOS_PATHS_LIST,number_of_frames=16,batch_size=50)
#print(len(next(c)))



