
# coding: utf-8

# In[1]:

import os
import tflearn
from skimage import color, io
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

get_ipython().magic(u'matplotlib inline')


# In[2]:

size_image =32

allX = np.zeros((100, size_image, size_image, 3), dtype='float32')
ally = np.zeros(100)
count = 0

#cat images and labels
for i in range(49):
        img = io.imread("cat/cat"+str(i+1)+".jpg")
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1

#dog images and labels
for i in range(51):
        img = io.imread("dog/dog"+str(i+1)+".jpg")
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1


print("number of images :", len(allX))
print("number of labels :", len(ally))
print("shape of image :", allX[20].shape)
print("all labels :", ally)


# In[3]:

plt.imshow(allX[90])


# In[4]:

#splitting dataset supaya random
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

print('number of training data :',len(X), 'number of training labels :', len(Y))
print('number of testing data :',len(X_test), 'number of testing labels :',len(Y_test))
print(Y_test)


# In[5]:

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

#print(Y)
print(Y_test)


# In[6]:

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

print(img_prep)
print(img_aug)


# In[8]:

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

#conv layer with filter sixe 3x3, number of filter 32 with activation relu
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')
#pooling layer with filter size 2x2
network = max_pool_2d(conv_1, 2)
#FC layer with number of neurons 512
network = fully_connected(network, 512, activation='relu')
#dropout layer to prevent overfitting
network = dropout(network, 0.5)
#final FC with classifier : softmax
network = fully_connected(network, 2, activation='softmax')

#calculate the accuracy
acc = Accuracy(name="Accuracy")
#define how network to be trained
#it uses crossentropy loss function, adam optimizer for gradient descent, learning rate 0.0005 
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')



# In[11]:

# Train using classifier
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=50, run_id='dogcat_cnn')


# In[ ]:




# In[ ]:



