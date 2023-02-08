import json
import keras
import scipy
import numpy as np
import math
import tensorflow as tf
import random

class Pix3DSequence(tf.keras.utils.Sequence):
    def __init__(self, data_json, shuffle=True, batch_size=32):
        self.dataset = []
        self.x = []
        self.y = []
        self.batch_size = batch_size
        
        with open(data_json) as file:
            json_data = json.load(file)

        for input in json_data:
            self.dataset.append(Pix3DData(input))

        if shuffle:
            random.shuffle(self.dataset)

        for data in self.dataset:
            self.x.append(data.img)
            self.y.append(data.voxel)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.array(keras.utils.img_to_array(keras.utils.load_img(file_name, target_size=(224,224)))) for file_name in batch_x]), np.array([np.array(scipy.io.loadmat(file_name)['voxel']) for file_name in batch_y])[::,::4,::4,::4]


class Pix3DDataset:
    def __init__(self):
        self.dataset = []
        with open('./data/pix3d/pix3d.json') as file:
            json_data = json.load(file)

        for input in json_data:
            self.dataset.append(Pix3DData(input))

    def generate_batch(self, batch_size=32, starting_index=0):
        images = []
        voxels = []
        for i in range(batch_size):
            data = self.dataset[starting_index + i]
            images.append(keras.utils.img_to_array(keras.utils.load_img(data.img, target_size=(224,224))))
            voxels.append(scipy.io.loadmat(data.voxel)['voxel'])

        return (np.array(images), np.array(voxels))


class Pix3DData:
    def __init__(self, input):
        self.img = './data/pix3d/' + input['img']
        self.category = input['category']
        self.img_size = input['img_size']
        self.two_d_keypoints = input['2d_keypoints']
        self.mask = './data/pix3d/' + input['mask']
        self.img_source = input['img_source']
        self.model = './data/pix3d/' + input['model']
        self.model_raw = input['model_raw']
        self.model_source = input['model_source']
        self.three_d_keypoints = './data/pix3d/' + input['3d_keypoints']
        self.voxel = './data/pix3d/' + input['voxel']
        self.rot_mat = input['rot_mat']
        self.trans_mat = input['trans_mat']
        self.focal_length = input['focal_length']
        self.cam_position = input['cam_position']
        self.inplane_rotation = input['inplane_rotation']
        self.truncated = input['truncated']
        self.occluded = input['occluded']
        self.slightly_occluded = input['slightly_occluded']
        self.bbox = input['bbox']

if __name__== "__main__":
    dataset = Pix3DDataset()

    images, voxels = dataset.generate_batch()

    print(images[0].shape)
    print(voxels[0].shape)

