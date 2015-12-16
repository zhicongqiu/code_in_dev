import glob,os,sys
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from img_crop import *
#from get_img_crop_array import get_img_crop_array

#path to image file
Vl_dict_train = {}
Vh_dict_train = {}
V_dict_test = {}
date_dict = {}
#test date
test_date = '2014-08-01'
#cropped ratio
cropped_ratio = 0.9
#training img file dir
trainL_img_dir = '/home/zhicong/Desktop/Train_low_rest/'
trainH_img_dir = '/home/zhicong/Desktop/Train_high_rest/'
#test img file dir
test_img_dir = '/home/zhicong/Desktop/Test_'+test_date+'/'
os.chdir('/home/zhicong/Desktop/summer_project_DNN/data/images/zipit/pictures/computed_truth/') 
for file in glob.glob('*.jpg'):
    #only use the file with .json label
    idx = file.find('2014')
    if os.path.isfile(file[:-3]+'json') and (idx is not -1):
        print file
        label_file = open(file[:-3]+'json','r')
        #split into training and testing set
        date_str = file[idx:idx+10]
        temp_dict = json.load(label_file)
        temp_occ = [temp_dict['occlusion']]*5
        v_num = temp_dict['vehicle_id'].upper()
        if v_num == ' V10':
            v_num = 'V10'
        temp_ori = [temp_dict['orientation'].upper()]*5
        if date_str == test_date:
            if v_num not in V_dict_test.keys():
                V_dict_test[v_num]=[[],[],[]] #2-dim list
            #append cropped images:[center,left,right,up,down]
            V_dict_test[v_num][0].extend(get_img_crop_array(file,cropped_ratio,test_img_dir))
            V_dict_test[v_num][1].extend(temp_occ)
            V_dict_test[v_num][2].extend(temp_ori)
        else:
            if temp_occ[0]<=0.2:
                if v_num not in Vl_dict_train.keys():
                    Vl_dict_train[v_num]=[[],[],[]] #2-dim list
                #append cropped images:[center,left,right,up,down]
                Vl_dict_train[v_num][0].extend(get_img_crop_array(file,cropped_ratio,trainL_img_dir))
                Vl_dict_train[v_num][1].extend(temp_occ)
                Vl_dict_train[v_num][2].extend(temp_ori)
            else:
                if v_num not in Vh_dict_train.keys():
                    Vh_dict_train[v_num]=[[],[],[]] #2-dim list 
                #append cropped images:[center,left,right,up,down]
                Vh_dict_train[v_num][0].extend(get_img_crop_array(file,cropped_ratio,trainH_img_dir))
                Vh_dict_train[v_num][1].extend(temp_occ)
                Vh_dict_train[v_num][2].extend(temp_ori)      
