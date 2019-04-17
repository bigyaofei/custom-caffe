import os
import random
import numpy as np
import cPickle
import cv2
import caffe

USE_GAUSS = True

ctg = ['negative','atypical','dysplasia','malignant','glandular','blood','artifacts','organisms','unknown']

class LandmarkData(caffe.Layer):
    def setup(self, bottom, top):
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        
        print("reading files from {}".format(self.param_str))
        self.filelist = []
	for line in open(self.param_str):
            self.filelist.append(line)
        random.shuffle(self.filelist)
        self.total = len(self.filelist)

        self.batch_size = 32
        self.height = 32 
        self.width = 32
        self.num_class = 7
        self.crop_width = self.width 
        self.crop_height = self.height
        self.index = 0
        
        self.down_sample_scale = 4 
        top[0].reshape(self.batch_size, 3, self.height, self.width)
        top[1].reshape(self.batch_size, self.num_class, 1, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for i in range(self.batch_size):
            image_str = self.filelist[self.index]
      
            label_str = image_str.split("_")[1][:-4]
	    
            self.index += 1
            self.index = self.index % self.total
		
            # read image and label
            img = cv2.imread(img_str)
            h,w,c = img.shape
            
            x = np.random.randint(0,10)
            y = np.random.randint(0,10)
            
            img_clip = img[x:x+32,y:y+32,:]

            label = np.array(ctg.index(label_str),dtype=np.float32)
            label = label.reshape((1,1,1))
            img_clip = np.array(img_clip, dtype=np.float32)
            img_clip -= self.mean
            img_clip /= 127.5

            top[0].data[i][...] = np.transpose(img_clip, (2,0,1))
            top[1].data[i][...] = label
