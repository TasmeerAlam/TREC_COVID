from keras.preprocessing.image import load_img
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.layers import UpSampling3D
from keras.layers import Conv3DTranspose
from keras.layers import Dropout
from keras.layers import Conv3D
from keras import layers
from keras.layers import Activation
from keras.layers import BatchNormalization
from glob import glob
from keras import backend as K
import tensorflow as tf
#tf.keras.optimizers

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
def decoder(input_shape):

	#inp1 = Input(shape = inp_shape_1)

	skip_layer_names = ['Mixed_5b','Mixed_4f','Mixed_4e','Mixed_4d','Mixed_4c','Mixed_4b','Mixed_3c','Mixed_3b','Conv3d_2c_3x3','Conv3d_1a_7x7']

	encoder = Inception_Inflated3d(include_top=False,input_shape=input_shape,weights='rgb_imagenet_and_kinetics')
	

	encoder_length = len(encoder.layers)
	inp1 = encoder.layers[-2].output
	
	inp2 = encoder.get_layer(skip_layer_names[0]).output

	#print(inp2.shape,'inp 2')


	inp3 = encoder.get_layer(skip_layer_names[1]).output

	#print(inp3.shape, 'inp 3')

	inp4 = encoder.get_layer(skip_layer_names[2]).output

	inp5 = encoder.get_layer(skip_layer_names[3]).output

	inp6 = encoder.get_layer(skip_layer_names[4]).output

	inp7 = encoder.get_layer(skip_layer_names[5]).output

	inp8 = encoder.get_layer(skip_layer_names[6]).output

	inp9 = encoder.get_layer(skip_layer_names[7]).output

	inp10 = encoder.get_layer(skip_layer_names[8]).output

	inp11 = encoder.get_layer(skip_layer_names[9]).output

	x = inp1

	
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
	#x = UpSampling3D(size=(2,2,2))(x)
	x = conv3d_bn(x, 384, 3, 3, 3, padding='same')

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


	x = layers.add([x,inp3])
	
	

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

	x = layers.add([x,inp4])
	
	
	branch_0 = conv3d_bn_Transpose(x, 128, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 128, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 256, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 24, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 64, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 64, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 64, 3, 3, 3, padding='same')


	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	x = layers.add([x,inp5])
	
	

	branch_0 = conv3d_bn_Transpose(x, 160, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 112, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 224, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 24, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 64, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 64, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 64, 3, 3, 3, padding='same')

	
	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)
	
	x = layers.add([x,inp6])
	
	
	branch_0 = conv3d_bn_Transpose(x, 192, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 96, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 208, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 16, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 48, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 64, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 64, 3, 3, 3, padding='same')


	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	x = layers.add([x,inp7])

	x = UpSampling3D(size=(2,2,2))(x)
	x = conv3d_bn(x, 192, 3, 3, 3, padding='same')


	branch_0 = conv3d_bn_Transpose(x, 128, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 128, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 192, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 32, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 96, 3, 3, 3, padding='same')

	branch_3 = UpSampling3D(size=(2,2,2))(x)
	branch_3 = conv3d_bn(x, 64, 3, 3, 3, padding='same')
	branch_3 = conv3d_bn_Transpose(x, 64, 3, 3, 3, padding='same')

	
	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	x = layers.add([x,inp8])
	
	


	branch_0 = conv3d_bn_Transpose(x, 64, 1, 1, 1, padding='same')

	branch_1 = conv3d_bn_Transpose(x, 96, 1, 1, 1, padding='same')
	branch_1 = conv3d_bn_Transpose(branch_1, 128, 3, 3, 3, padding='same')

	branch_2 = conv3d_bn_Transpose(x, 16, 1, 1, 1, padding='same')
	branch_2 = conv3d_bn_Transpose(branch_2, 32, 3, 3, 3, padding='same')

	branch_3 = conv3d_bn_Transpose(x, 32, 3, 3, 3, padding='same')

	x = layers.concatenate(
		[branch_0, branch_1, branch_2, branch_3],
		axis=4)

	x = layers.add([x,inp9])

	#x = layers.add([x,inp4])
	
	
	x = UpSampling3D(size=(1,2,2))(x)
	x = conv3d_bn(x, 192, 1, 3, 3, padding='same')

	x = layers.add([x,inp10])
	
	x = conv3d_bn_Transpose(x, 64, 3, 3, 3, strides=(1, 1, 1), padding='same')

	x = conv3d_bn_Transpose(x, 64, 1, 1, 1, strides=(1,1,1), padding='same')

	x = UpSampling3D(size=(1,2,2))(x)
	x = conv3d_bn(x, 64, 1, 3, 3, padding='same')

	x = layers.add([x,inp11])

	x = conv3d_bn_Transpose(x, 64, 7, 7, 7, strides=(2,2,2), padding='same')

	
	
	#x = conv3d_bn(x, 64, 3, 3, 3, strides=(2,1,1), padding='same')

	#x = conv3d_bn_Transpose(x, 2, 3, 3, 3, strides=(1,2,2), padding='same')

	outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)



	model = Model(inputs = [encoder.input], outputs = [outputs])

	for i in range(encoder_length-1):
		model.layers[i].trainable = False

	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-10))

	print(model.summary())

	return model




#decoder(input_shape=(16,224,224,3))