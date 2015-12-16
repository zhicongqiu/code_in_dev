import glob,os,sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import json
import cv2
from gmm_module import *
save_crop_dir =  'Train_high_inference/'
save_low_original = 'Train_low_original/'
save_high_original = 'Train_high_original/'
resize_x = 300
resize_y = 150
GMM_patch = {}
M_max = 30
#load a training file
Vl = pickle.load(open('pickle_file/resize_100/save_Vl_dict_train_ori.p','rb'))

#available vehicles used for training
available_list = ['V7']


for v_num in available_list:
    for v_img in xrange(len(Vl[v_num][0])):
        v_resized = cv2.resize(Vl[v_num][0][v_img],(resize_x,resize_y))
        cv2.imwrite(save_low_original+v_num+'_'+str(v_img)+'.jpg',v_resized)


            

