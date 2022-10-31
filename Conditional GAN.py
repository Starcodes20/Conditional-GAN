#!/usr/bin/env python
# coding: utf-8

# In[4]:


import wandb
wandb.init()


# In[5]:


import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import asarray
from keras.models import load_model


# In[6]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[7]:


tf.test.is_gpu_available(
  cuda_only=False, min_cuda_compute_capability=None
)


# In[30]:


from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
#from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate
from keras.activations import tanh, sigmoid
from keras.losses import binary_crossentropy


# In[92]:


CIFAR_DIR = r'C:/Users/AYO IGE/Downloads/cifar-10-python/cifar-10-batches-py/'


# In[93]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


# In[97]:


dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0,1,2,3,4,5,6]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)


# In[98]:


type(all_data)


# In[100]:


batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


# In[101]:


batch_meta


# In[102]:


data_batch1


# In[106]:


import numpy as np


# In[116]:


X = data_batch2[b'data']


# In[117]:


X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('uint8')


# In[118]:


X[0].max()


# In[119]:


plt.imshow(X[0])


# In[120]:


import matplotlib.pyplot as plt


# In[121]:


for i in range(25):
    plt.subplot(5,5,1 +i)
    plt.axis('off')
    plt.imshow(X[i])
plt.show()


# In[112]:





# In[ ]:





# In[ ]:





# In[12]:


def define_discriminator(in_shape=(32,32,3), n_classes = 10):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    in_image = Input(shape=in_shape)
    merge = Concatenate()([in_image,li])
    fe = Conv2D(128, (3,3), strides = (2,2), padding = 'same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3,3), strides = (2,2), padding = 'same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)
    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[13]:


test_discr = define_discriminator()
print(test_discr.summary())


# In[14]:


def define_generator(latent_dim, n_classes = 10):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 8*8
    li = Dense(n_nodes)(li)
    li = Reshape((8,8,1))(li)
    in_lat = Input(shape=(latent_dim,))
    
    n_nodes = 8192
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8,8,128))(gen)
    
    merge = Concatenate()([gen,li])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides = (2,2), padding= 'same')(gen)
    gen = LeakyReLU(alpha= 0.2)(gen)
    
    out_layer = Conv2D(3,(8,8), activation = 'tanh', padding = 'same')(gen)
    
    model = Model([in_lat, in_label], out_layer)
    return model


# In[15]:


test_gen = define_generator(100, n_classes=10)
print(test_gen.summary())


# In[16]:


def define_gan(g_model, d_model):
    d_model.trainable = False
    ##Connect generator and discriminator
    ## Add noise and label inputs
    gen_noise, gen_label = g_model.input
    ##get image output from generator
    gen_output = g_model.output
    
    #generator image output and correspnding input label
    gan_output = d_model([gen_output, gen_label])
    #define GAn model as taking noise and labels as inputs and outpuyying class and fake image
    model = Model([gen_noise, gen_label], gan_output)
    #compile
    opt= Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss= 'binary_crossentropy', optimizer = opt)
    return model


# In[17]:


def load_real_samples():
    (trainX, trainy), (_, _) = load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainy]


# In[18]:


def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y


# In[19]:


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes,n_samples)
    return [z_input, labels]


# In[20]:


def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    ## predict outputs
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# In[21]:


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs = 100, n_batch = 128):
    bat_per_epo = int(dataset[0].shape[0]/n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _= d_model.train_on_batch([X_real, labels_real], y_real)
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f'%
                 (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
    
    g_model.save('cfar_conditional_generator_250epochs.h5')


# In[22]:


from keras.metrics import accuracy


# In[23]:


latent_dim = 100


# In[24]:


d_model = define_discriminator()


# In[25]:


g_model = define_generator(latent_dim)


# In[26]:


gan_model = define_gan(g_model, d_model)


# In[27]:


dataset = load_real_samples()


# In[28]:


train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=250)


# In[ ]:


model = load_model('cfar_conditional_generator_250epochs.h5')


# In[ ]:


from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np


# In[ ]:


## generate multiple images
latent_points, labels = generate_latent_points(100,100)
labels = asarray([x for _ in range(10) for x in range(10)])
X = model.predict([latent_points, labels])
X = (X + 1)/2.0
X = (X * 255). astype(np.uint8)


# In[ ]:


def show_plot(examples, n):
    for i in range( n*n):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()


# In[ ]:


show_plot(X, 10)


# In[ ]:




