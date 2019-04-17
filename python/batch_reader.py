#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math
import signal
import random
import time
from multiprocessing import Process, Queue, Event

from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper

exitEvent = Event() # for noitfy all process exit.

#def handler(sig_num, stack_frame):
#    global exitEvent
#    exitEvent.set()
#signal.signal(signal.SIGINT, handler)


#############yao fei###############
def rotateWithLandmark(image, cx,cy,landmark_x,landmark_y, angle, scale, pointnum):
    if angle == 0:
        rot_image = image.copy()
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        return rot_image,landmark_x1,landmark_y1
    else:
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
        #rotate
  
        rot_image = cv2.warpAffine(image,M,(w,h))
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        for i in range(pointnum):
            landmark_x1[i] = M[0][0]*landmark_x[i]+M[0][1]*landmark_y1[i]+M[0][2]
            landmark_y1[i] = M[1][0]*landmark_x[i]+M[1][1]*landmark_y1[i]+M[1][2]
    return rot_image, landmark_x1,landmark_y1


def random_rotate(image,landmark_x,landmark_y,pointnum):
    img = image.copy()
    h,w,c = img.shape
    cx_delta = random.randint(0,15)
    cy_delta = random.randint(0,15)
    plusminus = random.randint(0,10)
    cx = w/2 
    cy = h/2
    if plusminus > 5:
        cx += cx_delta
        cy += cy_delta
    else:
        cx -= cx_delta
        cy -= cy_delta
    angle = random.randint(-20,20)
    scale = float(random.randint(7,11))/10
    rot_image, landmark_x1,landmark_y1 = rotateWithLandmark(image, cx,cy,landmark_x,landmark_y, angle, scale, pointnum)
    return rot_image, landmark_x1,landmark_y1


def random_SaltAndPepper(src):
    rn = random.randint(0,3)
    percetage = float(rn)/10  
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
    for i in range(SP_NoiseNum): 
        randR=np.random.randint(0,src.shape[0]-1) 
        randG=np.random.randint(0,src.shape[1]-1) 
        randB=np.random.randint(0,3)
        if random.randint(0,1)==0: 
            SP_NoiseImg[randR,randG,randB]=0 
        else: 
            SP_NoiseImg[randR,randG,randB]=255 
    return SP_NoiseImg 


def random_addGaussianNoise(image): 
    rn = random.randint(0,2)
    percetage = float(rn)/10
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0] 
    return G_Noiseimg

def random_darker(image):
    rn = random.randint(6,10)
    percetage = float(rn)/10
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

def random_brighter(image):
    rn = random.randint(10,20)
    percetage = float(rn)/10
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy


def random_blur(image):
    img = image.copy()
    k = [1,3,5,7,9,11]
    ks = random.sample(k,1)[0]
    img = cv2.GaussianBlur(img, (ks,ks), 0)
    return img




def faceaug(image,landmarkp,pointnum):
    landmark_x = []
    landmark_y = []

    for i in range(pointnum):
        landmark_x.append(landmarkp[i,0])
        landmark_y.append(landmarkp[i,1])       
    
    rd = np.random.randint(0,10)
    if rd<4:
        image = random_darker(image)
    elif rd >3 and rd<8:
        image = random_brighter(image)
    else:
        pass

    rd = np.random.randint(0,10)
    if rd<4:
        image = random_SaltAndPepper(image)
    elif rd >3 and rd<8:
        image = random_addGaussianNoise(image)
    else:
        pass
    
    rd = np.random.randint(0,10)
    if rd>5:
        image = random_blur(image)
    
    label = landmarkp
    rd = np.random.randint(0,10)
    if rd>5:
        image, landmark_x1,landmark_y1 = random_rotate(image,landmark_x,landmark_y,pointnum)
        landmarkr = []
        for i in range(pointnum):
            landmarkr.append([landmark_x1[i],landmark_y1[i]])
        label = np.array(landmarkr,dtype=np.float32)
    return image,label




############################


class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._process_list = []
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=2)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        self._max_epoch = kwargs['max_epoch']
        # start buffering
        self._start_buffering(kwargs['input_paths'], kwargs['landmark_type'])
      
    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    image_list, landmarks_list = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield image_list, landmarks_list

    def get_epoch(self):
        return self._curr_epoch

    def should_stop(self):
        if exitEvent.is_set() or self._curr_epoch > self._max_epoch:
            exitEvent.set()
            self.__clear_and_exit()
            return True
        else:
            return False

    def __clear_and_exit(self):
        print ("[Exiting] Clear all queue.")
        while True:
            time.sleep(1)
            _alive = False
            for i in range(self._process_num):
                try:
                    self._output_queue[i].get(block=True, timeout=0.01)
                    _alive = True
                except Exception as ex:
                    pass
            if _alive == False: break
        print ("[Exiting] Confirm all process is exited.")
        for i in range(self._process_num):
            if self._process_list[i].is_alive():
                print ("[Exiting] Force to terminate process %d"%(i))
                self._process_list[i].terminate()
        print ("[Exiting] Batch reader clear done!")

    def _start_buffering(self, input_paths, landmark_type):
        if type(input_paths) in [str, unicode]:
            input_paths = [input_paths]
        for input_path in input_paths:
            for line in open(input_path):
                info = LandmarkHelper.parse(line, landmark_type)
                self._sample_list.append(info)
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))
        #num_per_process = 2
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            self._process_list.append(p)

    def _process(self, idx, sample_list):
        __landmark_augment = LandmarkAugment()

        # read all image to memory to speed up!
        if self._kwargs['buffer2memory']:
            print ("[Process %d] Start to read image to memory! Count=%d"%(idx, len(sample_list)))
            sample_list = __landmark_augment.mini_crop_by_landmarks(sample_list, 4.5, self._kwargs['img_format'])
            print ("[Process %d] Read all image to memory finish!"%(idx))
        sample_cnt = 0 # count for one batch
        image_list, landmarks_list = [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                image = sample[0]
                landmarks = sample[1].copy()# keep deep copy
                #scale_range = (2.7, 3.3)
                #image_new, landmarks_new = __landmark_augment.augment(image, landmarks, self._kwargs['img_size'],
                                            #self._kwargs['max_angle'], scale_range)

                image,label = faceaug(image,landmarks,68)
                h,w,c = image.shape         
                scale = np.array((w,h),dtype=np.float32)
                
                landmarks_new = (label/scale).reshape((136,1,1))


                #for i in range(68):
                 #   x = int(w*landmarks_new[2*i])
                 #   y = int(h*landmarks_new[2*i+1])
                 #   cv2.circle(image,(x,y),2,(0,255,0),-1)
                #cv2.imwrite("/home/fyao/yaofei/face/yfnet/face.jpg",image)
            #print ("write ok!")
            #cv2.waitKey(0)


                im_ = cv2.resize(image,(224,224))                

                # caffe data format
                im_ = im_.astype(np.float32)
                im_ -= self.mean
                im_ /=127.5
                im_ = np.transpose(im_, (2,0,1))
                # sent a batch
                sample_cnt += 1
                image_list.append(im_)
                landmarks_list.append(landmarks_new)
                if sample_cnt >= self._kwargs['batch_size']:
                    self._output_queue[idx].put((np.array(image_list), np.array(landmarks_list)),timeout=0.01)
                    sample_cnt = 0
                    image_list, landmarks_list = [], []
                # if exit
                if exitEvent.is_set():
                    break
            if exitEvent.is_set():
                break
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0


