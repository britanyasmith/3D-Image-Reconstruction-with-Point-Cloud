import keras
import json
import numpy as np
import cv2
import os

NUM_VIEWS = 24
images = []
HEIGHT = 128
WIDTH = 128
PAD = 35

class ShapenetDataset(keras.utils.Sequence):
    def __init__(self, data_dir_imgs, data_dir_pcl, models, cats, numpoints=1024, variety=False, batch_size=32):
        self.data_dir_imgs = data_dir_imgs
        self.data_dir_pcl = data_dir_pcl
        self.models = models
        self.modelnames = []
        self.size = 0
        self.numpoints = numpoints
        self.variety = variety
        self.batch_size = batch_size

        for cat in cats:
            for filename in self.models[cat]:
                for i in range(NUM_VIEWS):
                    self.size = self.size + 1
                    self.modelnames.append(filename)


    def get_single_item(self, index):
        img_path = self.data_dir_imgs + self.modelnames[index] + '/rendering/' + (str(int(index % NUM_VIEWS)).zfill(2) + '.png')
        image = cv2.imread(img_path)[4:-5, 4:-5, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pcl_path = self.data_dir_pcl + self.modelnames[index] + '/pointcloud_' + str(self.numpoints) + '.npy'
        pcl_gt = np.load(pcl_path)

        if self.variety == True:
            metadata_path = self.data_dir_imgs + self.modelnames[index] + '/rendering/rendering_metadata.txt'
            metadata = np.loadtxt(metadata_path)
            x = metadata[(int(index % NUM_VIEWS))][0]
            xangle = np.pi / 180. * x
            y = metadata[(int(index % NUM_VIEWS))][1]
            yangle = np.pi / 180. * y
            return image, pcl_gt, xangle, yangle
        
        return image, pcl_gt

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        for idx in range(self.batch_size):
            if idx+index >= self.size:
                # Hack, just bind the item to the last input item we have.
                x, y = self.get_single_item(self.size - 1)
            else:
                x, y = self.get_single_item(index+idx)

            batch_x.append(x)
            batch_y.append(y)

        return np.asarray(batch_x), np.asarray(batch_y)
        
    def __len__(self):
        return self.size


