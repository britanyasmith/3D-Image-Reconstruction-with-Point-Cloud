import tensorflow as tf
import json
import keras
from keras.models import Model
from keras.applications import ResNet50
from keras.layers import Input, Dense, Lambda, LeakyReLU, Reshape
#from pix3d_dataset import Pix3DDataset, Pix3DSequence
from loss import batch_cd_loss
from shapenet_dataset import ShapenetDataset
import numpy
import math

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sesson = tf.compat.v1.Session(config=config)

# Create ResNet-50 Model for 2D feature extraction.
resnet50 = ResNet50(include_top=False, input_shape=(128,128,3), pooling='avg')

# Encode the 2D features extracted by RestNet-50.
def sample(args):
    mean, var = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean), mean=0.0, stddev=1.0)
    return mean + keras.backend.exp(0.5 * var) * epsilon


encoder_inputs = Input(shape=(100,), name='input')
mean = Dense(100, name='mean')(encoder_inputs)
sigma = Dense(100, name='sigma')(encoder_inputs)
z_prime = Lambda(sample, (100,), name='z_prime')([mean, sigma])

encoder = Model(inputs=encoder_inputs, outputs=z_prime, name='encoder')

# Decode and generate a Point Cloud.
NUM_POINTS = 1024
decoder_inputs = Input(shape=(100,), name='input')
x = Dense(512, name='hidden_layer1')(decoder_inputs)
x = LeakyReLU(alpha=0.5)(x)
x = Dense(1024, name='hidden_layer2')(x)
x = LeakyReLU(alpha=0.5)(x)
x = Dense(NUM_POINTS*3, activation='tanh', name='output')(x)
output = Reshape((NUM_POINTS,3))(x)
point_cloud_generation = Model(inputs=decoder_inputs, outputs=output, name='point_cloud_generation')

reconstnet = keras.Sequential()
reconstnet.add(resnet50)
reconstnet.add(Dense(100))
reconstnet.add(encoder)
reconstnet.add(point_cloud_generation)

# Define optimizer and loss function.
def loss_function(x, y):
    return batch_cd_loss(x,y)

reconstnet.compile(optimizer='adam', loss=loss_function, run_eagerly=True)
#reconstnet.compile(optimizer='adam', loss='mse', run_eagerly=True)

# Open the training/validation data split files.
with open('data/splits/train_models.json', 'r') as f:
    train_models_dict = json.load(f)

with open('data/splits/val_models.json', 'r') as f:
    val_models_dict = json.load(f)

data_dir_imgs = 'data/shapenet/ShapeNetRendering/'
data_dir_pcl = 'data/shapenet/ShapeNet_pointclouds/'

# Create our ShapenetDataset instance, which will help the model iterate over the dataset.
cats = []
cats.append('02691156')
shapenet_dataset = ShapenetDataset(data_dir_imgs, data_dir_pcl, train_models_dict, cats)

# Train the model.
reconstnet.fit(shapenet_dataset)

reconstnet.save('model')
