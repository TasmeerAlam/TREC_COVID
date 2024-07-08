from tf.keras.preprocessing.image import load_img
import numpy as np
from tf.keras.optimizers import Adam
from tf.keras.layers import Input
from tf.keras.models import Model
from tf.keras.layers import UpSampling3D
from tf.keras.layers import Conv3DTranspose
from tf.keras.layers import Dropout
from tf.keras.layers import Conv3D
from tf.keras import layers
from tf.keras.layers import Activation
from tf.keras.layers import BatchNormalization
from glob import glob
from tf.keras import backend as K
import tensorflow as tf

from i3d import Inception_Inflated3d

def conv3d_bn_Transpose(x,
			  filters,
			  num_frames,
			  num_row,
			  num_col,
			  padding='same',
			  strides=(1, 1, 1),
			  use_bias = False,
			  use_activation_fn = True,
			  use_bn = True,
			  name=None):
	#model_ = files.upload()
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv3DTranspose(
		filters, (num_frames, num_row, num_col),
		strides=strides,
		padding=padding,
		use_bias=use_bias,
		name=conv_name)(x)

	x = BatchNormalization(axis=4, scale=False, name=bn_name)(x)

	if use_activation_fn:
		x = Activation('relu', name=name)(x)

	return x

def conv3d_bn(x,
			  filters,
			  num_frames,
			  num_row,
			  num_col,
			  padding='same',
			  strides=(1, 1, 1),
			  use_bias = False,
			  use_activation_fn = True,
			  use_bn = True,
			  name=None):
	#model_ = files.upload()
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv3D(
		filters, (num_frames, num_row, num_col),
		strides=strides,
		padding=padding,
		use_bias=use_bias,
		name=conv_name)(x)

	
	x = BatchNormalization(axis=4, scale=False, name=bn_name)(x)

	if use_activation_fn:
		x = Activation('relu', name=name)(x)

	return x
def decoder(inp_shape_1, inp_shape_2, inp_shape_3, inp_shape_4, inp_shape_5):

	inp1 = Input(shape = inp_shape_1)

	inp2 = Input(shape = inp_shape_2)

	inp3 = Input(shape = inp_shape_3)

	inp4 = Input(shape = inp_shape_4)

	inp5 = Input(shape = inp_shape_5)

	x = inp1

	print(x.shape)

	x = UpSampling3D(size=(2,2,2))(x)
	x = conv3d_bn_Transpose(x, 256, 2, 7, 7, padding='same')

	
	# Mixed 5b
	branch_0 = conv3d_bn_Transpose(x, 256, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 160, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 320, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 32, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 128, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 128, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 128, 3, 3, 3, padding='same')

	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	x = layers.add([x,inp2])

	x = UpSampling3D(size=(2,2,2))(x)
	x = conv3d_bn(x, 384, 3, 3, 3, padding='same')
	
	

	branch_0 = conv3d_bn_Transpose(x, 112, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 144, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 288, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 32, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 64, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 64, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 64, 3, 3, 3, padding='same')

	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	
	
	branch_0 = conv3d_bn_Transpose(x, 112, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 144, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 288, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 32, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 64, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 48, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 48, 3, 3, 3, padding='same')


	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)
	
	

	branch_0 = conv3d_bn_Transpose(x, 112, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 144, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 288, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 32, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 64, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 48, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 48, 3, 3, 3, padding='same')

	
	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)
	
	x = layers.add([x,inp3])
	
	

	branch_0 = conv3d_bn_Transpose(x, 192, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 144, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 208, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 16, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 48, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 32, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 32, 3, 3, 3, padding='same')

	
	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)
	
	

	x = UpSampling3D(size=(2,2,2))(x)
	x = conv3d_bn(x, 192, 3, 3, 3, padding='same')
	
	a_ = x.shape



	branch_0 = conv3d_bn_Transpose(x, 64, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 48, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 128, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 8, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 32, 3, 3, 3, padding='same')

	branch_3 = conv3d_bn_Transpose(x, 32, 3, 3, 3, padding='same')

	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	a = x.shape

	x = layers.add([x,inp4])
	
	
	x = UpSampling3D(size=(1,2,2))(x)
	x = conv3d_bn(x, 192, 1, 3, 3, padding='same')
	
	x = conv3d_bn_Transpose(x, 64, 3, 3, 3, strides=(1, 1, 1), padding='same')

	x = conv3d_bn_Transpose(x, 64, 1, 1, 1, strides=(1,1,1), padding='same')

	x = UpSampling3D(size=(1,2,2))(x)
	x = conv3d_bn(x, 64, 1, 3, 3, padding='same')

	x = layers.add([x,inp5])

	x = conv3d_bn_Transpose(x, 64, 7, 7, 7, strides=(2,2,2), padding='same')

	
	
	#x = conv3d_bn(x, 64, 3, 3, 3, strides=(2,1,1), padding='same')

	#x = conv3d_bn_Transpose(x, 2, 3, 3, 3, strides=(1,2,2), padding='same')

	outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)

	model = Model(inputs = [inp1,inp2, inp3, inp4, inp5], outputs = [outputs])


	return model

skip_layer_names = ['Mixed_5b','Mixed_4d','Mixed_3b','Conv3d_1a_7x7']
encoder = Inception_Inflated3d(include_top=False,input_shape=(16,224,224,3),weights='rgb_imagenet_and_kinetics')
inp1_shape = encoder.layers[-2].output.shape
inp2_shape = encoder.get_layer(skip_layer_names[0]).output.shape
inp3_shape = encoder.get_layer(skip_layer_names[1]).output.shape
inp4_shape = encoder.get_layer(skip_layer_names[2]).output.shape
inp5_shape = encoder.get_layer(skip_layer_names[3]).output.shape

decoder(inp1_shape,inp2_shape,inp3_shape,inp4_shape,inp5_shape)