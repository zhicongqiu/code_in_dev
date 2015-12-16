import numpy as np
import cv2
def img_crop(file,ratio,what,save_crop_dir):
    img = cv2.imread(file)
    #crop 6%
    w,h,d = img.shape
    img = img[np.floor(w*0.03):np.floor(w*0.97),
                      np.floor(h*0.03):np.floor(h*0.97)]
    w,h,d = img.shape
    if what == 'center':
        cropped_img = img[np.floor(w*((1-ratio)/2)):np.floor(w*(ratio+(1-ratio)/2)),
                          np.floor(h*((1-ratio)/2)):np.floor(h*(ratio+(1-ratio)/2))]
    elif what == 'up':
        cropped_img = img[0:np.floor(w*ratio),
                          np.floor(h*((1-ratio)/2)):np.floor(h*(ratio+(1-ratio)/2))]
    elif what == 'down':
        cropped_img = img[np.ceil(w*(1-ratio)):,
                          np.floor(h*((1-ratio)/2)):np.floor(h*(ratio+(1-ratio)/2))]
    elif what == 'right':
        cropped_img = img[np.floor(w*((1-ratio)/2)):np.floor(w*(ratio+(1-ratio)/2)),
                          np.ceil(h*(1-ratio)):]
                           
    elif what == 'left':
        cropped_img = img[np.floor(w*((1-ratio)/2)):np.floor(w*(ratio+(1-ratio)/2)),
                          0:np.floor(h*ratio)]
    else:
        'not recognized...'
    #save the image?
    if save_crop_dir is not 'none':
        #cropped_img.save(save_crop_dir+file)
        cv2.imwrite(save_crop_dir+file,cropped_img)
    return cropped_img

def get_img_crop_array(input_img,ratio,save_crop_dir):
    
    if save_crop_dir[-1] == '/':
        return [img_crop(input_img,ratio,'center',save_crop_dir+'center_'),
                img_crop(input_img,ratio,'left',save_crop_dir+'left_'),
                img_crop(input_img,ratio,'right',save_crop_dir+'right_'),
                img_crop(input_img,ratio,'up',save_crop_dir+'up_'),
                img_crop(input_img,ratio,'down',save_crop_dir+'down_')]
    else:
        return [img_crop(input_img,ratio,'center',save_crop_dir+'/center_'),
                img_crop(input_img,ratio,'left',save_crop_dir+'/left_'),
                img_crop(input_img,ratio,'right',save_crop_dir+'/right_'),
                img_crop(input_img,ratio,'up',save_crop_dir+'/up_'),
                img_crop(input_img,ratio,'down',save_crop_dir+'/down_')]
        
