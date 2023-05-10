import numpy as np
import torch
import cv2
import argparse
import sys
import tensorflow as tf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def run_model(device='none', data='segmentation/images/food_tray.jpg'):
    image = cv2.imread(data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sys.path.append("..")
    
    sam_checkpoint = "segmentation/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # device = torch.device("mps") # Apple M# Metal
    # device = 'cuda' # CUDA
    # if no device, set to 0
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    if(device == 'cuda'):
        device = 'cuda'
        sam.to(device=device)
        print('device loaded:', device)
    elif(device == 'mps'):
        device = torch.device('mps')
        sam.to(device=device)
        print('device loaded:', device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
 
    '''
    #Filtering on size of mask
    big_filter = 20000
    big_masks = list(filter(lambda mask: mask['area'] > big_filter, masks))
    '''

    predicted_masks = list(filter(lambda mask: (mask['predicted_iou'] > 1.0 and mask['stability_score'] > 0.97), masks))
    big_boxes = []
    cords = []
    #creates black bounding boxes around masks, instead of having adjacent parts of the image
    for i in predicted_masks[1:6]:  
        mask_image = i['segmentation'].astype(np.uint8) * 255
        resultimage = cv2.bitwise_and(image, image, mask=mask_image)
        box = get_bounding_box(i, resultimage) #includes adjacent parts of image in bounding boxes
        big_boxes.append(box)
        cord = get_coordinates(i)
        cords.append(cord)

    big_boxes2 = [get_bounding_box(mask, image) for mask in predicted_masks[1:6]]

    # WHITE PADDING
    # cropped_output1 = np.array([B2W(tf.image.resize_with_crop_or_pad(box, 224, 224)) for box in big_boxes]) 
    # cropped_output2 = np.array([B2W(tf.image.resize_with_crop_or_pad(box, 224, 224)) for box in big_boxes2])
    # # BLACK PADDING
    cropped_output1 = np.array([tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes]) 
    cropped_output2 = np.array([tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes2])

    return cropped_output1, cropped_output2, cords, data

def get_bounding_box(mask, img):
        XYWH = mask['bbox']
        return img[XYWH[1]:XYWH[1]+XYWH[3], XYWH[0]:XYWH[0]+XYWH[2]]

def get_coordinates(mask):
    XYWH = mask['bbox']
    return [XYWH[0], XYWH[1], XYWH[2], XYWH[3]]
    
def B2W(img):
    black_pixels = np.where(
            (img[:, :, 0] == 0) & 
            (img[:, :, 1] == 0) & 
            (img[:, :, 2] == 0)
        )
    img = img.numpy()
    img[black_pixels] = [255,255,255]
    img = tf.convert_to_tensor(img)
    return img  
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', '--device', choices=['none', 'mps', 'cuda'], 
                        default='cuda', help='Either none, mps, cuda')
    parser.add_argument('-d', '--data', 
                        default='images/food_tray.jpg', help='Path to image to segment')
    
    args = parser.parse_args()
    print('args:', args)

    run_model(device=args.device, 
              data=args.data)