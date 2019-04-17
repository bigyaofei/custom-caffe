import os
import random
import numpy as np
import cPickle
import cv2
import caffe
import time

image_path = "/home/fyao/data/..."

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
        landmark_x.append(landmarkp[2*i])
        landmark_y.append(landmarkp[2*i+1])       
    
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
            landmarkr.append(landmark_x1[i])
            landmarkr.append(landmark_y1[i])
        label = np.array(landmarkr,dtype=np.float32)
    return image,label


class LandmarkData(caffe.Layer):
    def setup(self, bottom, top):
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        
        print("reading files from {}".format(self.param_str))
        self.filelist = []
	for line in open(self.param_str):
            self.filelist.append(line.strip())
        random.shuffle(self.filelist)
        self.total = len(self.filelist)

        self.batch_size = 64
        self.height = 224 
        self.width = 224
        self.channel = 3
        self.num_point = 68
        self.crop_width = self.width 
        self.crop_height = self.height
        self.index = 0
        
        top[0].reshape(self.batch_size, self.channel, self.height, self.width)
        top[1].reshape(self.batch_size, self.num_point*2, 1, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        starttime = time.time()
        for i in range(self.batch_size):
            image_label = self.filelist[self.index]
            img_str = image_label.split(" ")[0]
            imgfile = img_str
            label_str = image_label.split(" ")[1:]
	    
            self.index += 1
            self.index = self.index % self.total

            # read image and label
            img = cv2.imread(imgfile)

 
            label = np.array(label_str,dtype=np.float32)

            img,label = faceaug(img,label,self.num_point)

            h,w,c = img.shape         
            scale = np.array((w,h),dtype=np.float32)
            label = ((label.reshape((self.num_point,2)))/scale).reshape(self.num_point*2,1,1)

            #for i in range(68):
               # x = int(w*label[2*i])
              #  y = int(h*label[2*i+1])
             #   cv2.circle(img,(x,y),2,(0,255,0),-1)
            #cv2.imwrite("/home/fyao/yaofei/face/face.jpg",img)
            #print ("write ok!")
            #cv2.waitKey(0)

               
            img = cv2.resize(img,(self.height,self.width))
            img = np.array(img, dtype=np.float32)
            img -= self.mean
            img /= 127.5
            img = np.transpose(img, (2,0,1))
         
            #print img.shape
            top[0].data[i][...] = img
            top[1].data[i][...] = label
        endtime = time.time() 
        print ("batch time cost : {}".format((endtime-starttime)))
