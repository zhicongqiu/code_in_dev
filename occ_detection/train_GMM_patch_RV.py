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
available_list = ['V1']
v_num = 'V1'
gmm_pval = np.zeros((resize_y*resize_x/25,len(Vl[v_num][0])-1))

for v_num in available_list:
    for v_img in xrange(len(Vl[v_num][0])):
        v_resized = cv2.resize(Vl[v_num][0][v_img],(resize_x,resize_y))
        cv2.imwrite(save_low_original+v_num+'_'+str(v_img)+'.jpg',v_resized)
        if v_img !=25: #exclude bad images
            #put 5*5*3 images in diff lists
            for k in range(1,resize_x/5+1):
                for kk in range(1,resize_y/5+1):
                    temp_key = str(kk-1)+'_'+str(k-1)
                    if temp_key not in GMM_patch.keys():
                        GMM_patch[temp_key] = v_resized[5*(kk-1):5*kk,5*(k-1):5*k].ravel()
                    else:
                        GMM_patch[temp_key] = np.vstack((GMM_patch[temp_key],
                                                         v_resized[5*(kk-1):5*kk,5*(k-1):5*k].ravel()))

print 'learning GMM model for each vehicle, for each patch\n'
v_num_GMM = {}
num_comp_hist={}

for v_num in available_list:
    #learn a GMM model on each 5*5 blocks
    count = 0
    for k in range(1,resize_x/5+1):
        for kk in range(1,resize_y/5+1): 
            temp_key = str(kk-1)+'_'+str(k-1)
            if v_num not in v_num_GMM.keys():
                v_num_GMM[v_num] = {}
                num_comp_hist[v_num] = np.zeros((resize_y*resize_x/25,1))
            num_comp_hist[v_num][count],v_num_GMM[v_num][temp_key],gmm_pval[count,:] = GMM_BIC(GMM_patch[temp_key],M_max)
            count+=1
del Vl

print 'save GMM models\n'
pickle.dump(v_num_GMM,open('pickle_file/v_num_GMM_V1_ori.p','wb'))


#make inference on the high occluded images and save modified figure
Vh = pickle.load(open('pickle_file/resize_100/save_Vh_dict_train_ori.p','rb'))
Vh_blackout = {}
p_val_list = np.ones((resize_y*resize_x/25,3))
for v_num in available_list:
    for v_img in range(len(Vh[v_num][0])):
        re_img = cv2.resize(Vh[v_num][0][v_img],(resize_x,resize_y))
        cv2.imwrite(save_high_original+v_num+'_'+str(v_img)+'.jpg',re_img)
        count = 0
        for k in range(1,resize_x/5+1):
            for kk in range(1,resize_y/5+1):
                temp_key = str(kk-1)+'_'+str(k-1)
                p_val_list[count][0] = v_num_GMM[v_num][temp_key].score(
                    re_img[5*(kk-1):5*kk,5*(k-1):5*k].ravel())
                p_val_list[count][1] = kk
                p_val_list[count][2] = k
                count+=1
        #sort p_val_list
        p_val_list = p_val_list[p_val_list[:,0].argsort()]
        temp_occ = Vh[v_num][1][v_img]
        covered_num = int(resize_y*resize_x/25*temp_occ)
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
pickle.dump(Vh_blackout,open('pickle_file/Vh_blackout_V1_ori.p','wb'))

