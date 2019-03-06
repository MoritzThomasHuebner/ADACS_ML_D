# Image classification demonstration with Keras
# Dr. Matthew Smith, Swinburne University of Technology, CAS / ADACS

# Before using this script, make sure you:
# (i) Create the train and test directories properly.
# You'll need: ./train/cats, ./train/dogs, ./test/cats, ./test/dogs, ./preview
# (ii) Extract the zip files into the correct location
# There will be 1000 training images each of cats and dogs, with 100 training images again for each.
# The preview folder, initially empty, will be filled by the Preview_Image_Generator() function.

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from utilities import *

# Investigate the imagedatagenerator
# This is not part of the actual ML process - you can
# comment this out after its first use
Preview_Image_Generator(True)

# Dimensions of our images.
img_width, img_height = 150, 150

train_dir = './train'
test_dir = './test'
N_train = 2000			# Total number of training files we have (1000+1000)
N_test = 200			# Total number of test files we have (100+100)
N_epochs = 1 			# This is not going to be enough.

# It is feasible that the amount of data we are dealing with is substantial.
# So, in this case, we will practice the use of generators.

# Prepare augmented training data generator
train_datagen = Prepare_Image_Data(0.2, 0.2, True, False)
test_datagen = Prepare_Image_Data(0.0, 0.0, False, True)

# Create the training and testing data generators.
train_generator = train_datagen.flow_from_directory(train_dir,
                  target_size=(img_width, img_height), batch_size=32,class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                 target_size=(img_width, img_height), batch_size=32,class_mode='binary')

# Set up our model
# Need to take care with the input shape configuration
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Create the model
model = Sequential()

# Add convolution and pooling layers to find features and reduce problem size.
model.add(Conv2D(32, (3, 3), input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce our problem to a one dimensional form
# After this, it is like we have a 1D sequence
# as we did in previous examples.
model.add(Flatten())

# From here it is business as usual.
# This is a binary classification problem, so make sure the output layer
# has one output. Don't place dropout on the output layer.
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Print out a summary of the model so we can see how much trouble we are in.
print(model.summary())

# Perform the actual training using fit_generator instead of fit
model.fit_generator(train_generator,
    steps_per_epoch= N_train // 32,
    epochs=N_epochs,
    validation_data=test_generator,
    validation_steps= N_test // 32)

# Keeping in-line with previous notes - let's use evaluate to check accuracy.
# Only this time, we are evaluating test data with a generator - use evaluate_generator
scores = model.evaluate_generator(test_generator,steps=N_test//32)
print("Accuracy: %.2f%%" % (scores[1]*100))


