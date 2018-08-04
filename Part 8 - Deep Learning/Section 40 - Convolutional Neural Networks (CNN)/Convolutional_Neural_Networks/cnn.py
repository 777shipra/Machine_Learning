# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#since the data preprocessing was done manually so 
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential#to initialize
from keras.layers import Convolution2D#for convolutional layers , as images are 2d unlike videos in 3d
from keras.layers import MaxPooling2D#pooling layers
from keras.layers import Flatten#for input layer 
from keras.layers import Dense#to add input layers to NN

# Initialising the CNN
#making an object of sequential
classifier = Sequential()

# Step 1 - Convolution
#it is the first layer of CNN
#32,3,3 no of filter used and each filter will create one feature map . 3X3 matrix rows and column of 32 feature detector
#working on CPU and also beginning number is 32
#input_shape ->shape of the input image 
#as all our images are not in the same size so force them into one format 64,64,3 is the 
# 3 is for the 3 channels of the color image -> blue green red channel
#64X64 format are enough for CPU 
#128X128 or 255X255 for GPU 
#for keras (64,64,3) is the format (dimention,dimention,channel)
#channel is 1 for black and white image 
#activation function just to sure we don't have any -ve pixels to take care of the non-linearity 
#rectifier function is the activation function 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#size of feature map is reduced to half in this step
#reducing in feature map reduce the set of nodes in fully connected layers 
#hence making model high computational and less time complexity
#size of the subtable to slide over the feature map , it is an 2X2 matrix (general)
#by 2X2 we don't lose the originality of the image 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#why don't we lose the originality of features by flatting
#because we already extracted the features by convolutional and pooling steps applied on the input image 
#flattening gives us one single vector  
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)