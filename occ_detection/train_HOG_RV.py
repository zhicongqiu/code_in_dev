import glob,os,sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from skimage.feature import hog
import json
import cv2
from gmm_module import *
save_crop_dir =  'Train_high_inference_HOG/'
block_size =10
resize_x = 300
resize_y = 150

#load a training file
Vl = pickle.load(open('pickle_file/save_Vl_dict_train_ori.p','rb'))
#available vehicles used for training
available_list = ['V1']
V1_clean = [7,10,15,17,18]
V1_HOG = [[] for i in range(len(V1_clean))]
M_max = 100
num_comp_hist = np.zeros((len(V1_clean),1))
for v_num in available_list:
    temp_count = 0
    for v_img in V1_clean:
        GMM_patch = []
        v_resized = cv2.resize(Vl[v_num][0][v_img],(resize_x,resize_y))
        #put block_size*block_size*3 images in diff lists
        for k in range(1,resize_x/block_size+1):
            for kk in range(1,resize_y/block_size+1):
                #turn into grey
                gray_image = cv2.cvtColor(v_resized[block_size*(kk-1):block_size*kk,
                                                    block_size*(k-1):block_size*k],
                                          cv2.COLOR_BGR2GRAY)
                #get hog features
                V1_HOG[temp_count].append(hog(gray_image,pixels_per_cell=(5, 5),
                                              cells_per_block=(1, 1)))
        temp_count+=1
del Vl

print 'save GMM models\n'
pickle.dump(V1_HOG,open('V1_clean_HOG.p','wb'))

#make inference on the high occluded images and save modified figure
Vh = pickle.load(open('pickle_file/save_Vh_dict_train_ori.p','rb'))
Vh_blackout = {}
dist_list = np.zeros((resize_x*resize_y/(block_size*block_size),3))
for v_num in available_list:
    for v_img in range(len(Vh[v_num][0])):
        re_img = cv2.resize(Vh[v_num][0][v_img],(resize_x,resize_y))
        count = 0
        for k in range(1,resize_x/block_size+1):
            for kk in range(1,resize_y/block_size+1):
                min_score = float('inf')
                temp_key = str(kk-1)+'_'+str(k-1)
                for prototype in V1_HOG:
                    #turn into grey
                    img_hog = hog(cv2.cvtColor(
                    re_img[block_size*(kk-1):block_size*kk,block_size*(k-1):block_size*k],
                        cv2.COLOR_BGR2GRAY),pixels_per_cell=(5, 5),cells_per_block=(1, 1))
                    #get the least deviation over all proto
                    temp = np.linalg.norm(img_hog-prototype[count])
                    if temp<min_score:
                        min_score = temp
                #get the minimal score under all the prototypes
                dist_list[count][0] = min_score
                dist_list[count][1] = kk
                dist_list[count][2] = k
                count+=1
        
        #sort p_val_list
        dist_list = dist_list[dist_list[:,0].argsort()]
        temp_occ = Vh[v_num][1][v_img]
        covered_num = int(resize_x*resize_y/(block_size*block_size)*temp_occ)
        #set the first covered_num_block to zeros
        for k in range(covered_num):
            d1 = dist_list[covered_num-1-k][1]
            d2 = dist_list[covered_num-1-k][2]
            re_img[block_size*(d1-1):block_size*d1,block_size*(d2-1):block_size*d2] = 0
        #save the modified image in a dictionary
        if v_num not in Vh_blackout.keys():
            Vh_blackout[v_num] = [[],[],[]]
        Vh_blackout[v_num][0].extend([re_img])
        Vh_blackout[v_num][1].extend([Vh[v_num][1][v_img]])
        Vh_blackout[v_num][2].extend([Vh[v_num][2][v_img]])
        #save the modified image in a file for visualization
        cv2.imwrite(save_crop_dir+v_num+'_'+str(v_img)+'.jpg',re_img)
pickle.dump(Vh_blackout,open('V1h_hog_blackout.p','wb'))
