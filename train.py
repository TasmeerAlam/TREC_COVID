from i3d_decoder import decoder
from generator import generator
import os
from glob import glob

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

model = decoder(input_shape=(16,224,224,3))

VIDEOS_PATHS = os.path.join('.','processed_data','*')

VIDEOS_PATHS_LIST = [videos for videos in glob(VIDEOS_PATHS)]

n = len(VIDEOS_PATHS_LIST)

train_paths = VIDEOS_PATHS_LIST[int(n*0.75):]
test_paths = VIDEOS_PATHS_LIST[:int(n*0.75)]

#BS = 10

train_gen = generator(video_paths=train_paths,number_of_frames=16,batch_size=1)
test_gen = generator(video_paths=test_paths,number_of_frames=16,batch_size=1)

H = model.fit_generator(
	train_gen, validation_data = test_gen,
	validation_steps=5,
	steps_per_epoch= 5,
	epochs=25)


#print(next(data_gen))