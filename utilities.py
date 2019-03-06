# Utilities.py
# Various functions for the Image Classification demonstration with Keras
# Dr. Matthew Smith, SUT, ADACS/CAS

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


def swish(x):
	# The swish activation function
	# Not currently used, but you *might* want to.
	beta = 1.5
	return beta*x*np.exp(x)/(np.exp(x)+1)

def Preview_Image_Generator(flip_mode):
	#  This function is designed to give us a feel for the behaviour
	# of the imagedatagenerator function shipped with Keras.
	# It will augment the data contained in 0.jpg (cats, 'cause cats are cool)
	# and produce variations of this image, saved in ./preview
	datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2, height_shift_range=0.2,
	                              shear_range=0.2, zoom_range=0.2,horizontal_flip=flip_mode,fill_mode='nearest')

	img = load_img('./train/cats/0.jpg')  # Work with image 0 to start with
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
		i += 1
		if i > 20:
			break  # otherwise the generator would loop indefinitely

	return 0


def Prepare_Image_Data(shear, zoom, flip_mode, test_flag):
	# Decide what features to use in ImageDataGenerator depending on
	# whether or not we are training or testing.
	if (test_flag == False):
		# Training data - will employ data augmentation, so can flip, zoom, shear etc.
		# Normalise the RGB data to between 0 and 1
		datagen = ImageDataGenerator(rescale=1.0/255.0,
	        	  shear_range = shear, zoom_range = zoom, horizontal_flip = flip_mode)
	else:
		# We don't need to augment the test data set - no reason to shift, zoom or flip.
		# Still need to rescale, though, since training employed normalisation.
		datagen = ImageDataGenerator(rescale=1.0/255.0)

	return datagen

