from img_crop import img_crop
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
