import glob,os,sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from skimage.feature import hog
import json
import cv2
from gmm_module import *
save_crop_dir =  'Train_high_inference_wholeGMM/'
resize_x = 300
resize_y = 150

#load a training file
Vl = pickle.load(open('pickle_file/save_Vl_dict_train_ori.p','rb'))
#available vehicles used for training
available_list = ['V1']
v_num_GMM = {}
V1_clean = [7,10,15,17,18]
M_max = 100
num_comp_hist = np.zeros((len(V1_clean),1))
count = 0
for v_num in available_list:
    temp_count = 0
    for v_img in V1_clean:
        temp_count+=1
        GMM_patch = []
        v_resized = cv2.resize(Vl[v_num][0][v_img],(resize_x,resize_y))
        #put 5*5*3 images in diff lists
        for k in range(1,resize_x/5+1):
            for kk in range(1,resize_y/5+1):
                #turn into grey
                #gray_image = cv2.cvtColor(v_resized[5*(kk-1):5*kk,5*(k-1):5*k],
                #                          cv2.COLOR_BGR2GRAY)
                if GMM_patch == []:
                    GMM_patch = v_resized[5*(kk-1):5*kk,5*(k-1):5*k].ravel()
                else:
                    GMM_patch = np.vstack((GMM_patch,
                                           v_resized[5*(kk-1):5*kk,5*(k-1):5*k].ravel()))
        print 'learning GMM model for each vehicle, for each patch\n'
        num_comp_hist[count],v_num_GMM[v_num+'_'+str(v_img)] = GMM_BIC(GMM_patch,M_max)
        count+=1
del Vl

print 'save GMM models\n'
pickle.dump(v_num_GMM,open('V1_clean_GMM.p','wb'))

#make inference on the high occluded images and save modified figure
Vh = pickle.load(open('pickle_file/save_Vh_dict_train_ori.p','rb'))
Vh_blackout = {}
p_val_list = np.ones((resize_x*resize_y/(5*5),3))
for v_num in available_list:
    for v_img in range(len(Vh[v_num][0])):
        re_img = cv2.resize(Vh[v_num][0][v_img],(resize_x,resize_y))
        count = 0
        for k in range(1,resize_x/5+1):
            for kk in range(1,resize_y/5+1):
                max_score = float('-inf')
                temp_key = str(kk-1)+'_'+str(k-1)
                for prototype in v_num_GMM:
                    temp = v_num_GMM[prototype].score(
                        re_img[5*(kk-1):5*kk,5*(k-1):5*k].ravel())
                    if temp>max_score:
                        max_score = temp
                #get the minimal score under all the prototypes
                p_val_list[count][0] = max_score
                p_val_list[count][1] = kk
                p_val_list[count][2] = k
                count+=1
        #sort p_val_list
        p_val_list = p_val_list[p_val_list[:,0].argsort()]
        temp_occ = Vh[v_num][1][v_img]
        covered_num = int(resize_x*resize_y/(5*5)*temp_occ)
        #set the first covered_num_block to zeros
        for k in range(covered_num):
            d1 = p_val_list[k][1]
            d2 = p_val_list[k][2]
            re_img[5*(d1-1):5*d1,5*(d2-1):5*d2] = 0
        #save the modified image in a dictionary
        if v_num not in Vh_blackout.keys():
            Vh_blackout[v_num] = [[],[],[]]
        Vh_blackout[v_num][0].extend([re_img])
        Vh_blackout[v_num][1].extend([Vh[v_num][1][v_img]])
        Vh_blackout[v_num][2].extend([Vh[v_num][2][v_img]])
        #save the modified image in a file for visualization
        cv2.imwrite(save_crop_dir+v_num+'_'+str(v_img)+'.jpg',re_img)
pickle.dump(Vh_blackout,open('V1h_blackout.p','wb'))
