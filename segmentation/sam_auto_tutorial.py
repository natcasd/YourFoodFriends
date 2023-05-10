import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import tensorflow as tf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mpl_toolkits.axes_grid1 import ImageGrid
import math

'''
# plotting fix lol
import tkinter
from matplotlib import use as mpl_use
mpl_use('TkAgg')
'''

def run_model(device='none', show_original='n', data='segmentation/images/food_tray.jpg', filter_method='big', show_bboxes_grid='n', show_masks='n', show_each='n'):
    print("model running")
    print(os.getcwd())
    print('data:', data)

    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            masked = np.dstack((img, m*0.35))
            ax.imshow(masked)
            # cutouts.append(ann['bbox']) #img gives just a solid block of color; 'segmentation' gives bicolor contrast image of mask; 'crop_box' gives...

    image = cv2.imread(data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('after reading image')
    if(show_original=='y'):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    
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

    print('generating mask')
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())

    if(show_masks=='y'):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 

    bounding_boxes = []
    def get_bounding_box(mask, image1=image):
        XYWH = mask['bbox']
        return image1[XYWH[1]:XYWH[1]+XYWH[3], XYWH[0]:XYWH[0]+XYWH[2]]

    def get_coordinates(mask):
        XYWH = mask['bbox']
        return [XYWH[1], XYWH[0], XYWH[3], XYWH[2]]
    
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
    
    print('creating bounding boxes')
    for mask in masks:
        prediction = mask['predicted_iou']
        stability = mask['stability_score']
        area = mask['area']

        bounding_box = get_bounding_box(mask)
        if(show_each=='y'):
            plt.imshow(bounding_box)
            plt.title(f'Confidence:{prediction}\nStability:{stability}\nArea:{area}')
            plt.show() #TODO: White out everything except mask
        bounding_boxes.append(bounding_box)

    

    # Below can be optimized into small if/else blocks
    if(filter_method=='stable'):
        stability_filter = 0.97
        filtered_stable_masks = filter(lambda mask : mask['stability_score'] > stability_filter, masks)
        stable_boxes = [get_bounding_box(mask) for mask in filtered_stable_masks]
        
        cropped_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in stable_boxes])
                        
        np.savez_compressed('cropped_stable_output', cropped_output)

    elif(filter_method=='big'):
        big_filter = 30000
        filtered_big_masks = filter(lambda mask: mask['area'] > big_filter, masks)
        big_boxes = [get_bounding_box(mask) for mask in filtered_big_masks]

        cropped_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes])

        np.savez_compressed('cropped_big_output', cropped_output)

    elif(filter_method=='boundingboxes'):
        big_filter = 20000
        max_area = 100000
        #filtered_big_masks = list(filter(lambda mask: (), masks))
        #[print(mask['predicted_iou']) for mask in filtered_big_masks]
        #[print(mask['stability_score']) for mask in filtered_big_masks]
        big_masks = list(filter(lambda mask: mask['area'] > big_filter, masks))
        predicted_masks = list(filter(lambda mask: mask['predicted_iou'] > 1.0, masks))
        stable_masks = list(filter(lambda mask: mask['stability_score'] > 0.97, predicted_masks))
        # stable_masks = list(filter(lambda mask: mask['area'] > big_filter, stable_masks))
        big_boxes = []
        cords = []
        # for i in masks:
        # for i in masks:
        # for i in predicted_masks:
        # for i in predicted_masks:
        for i in stable_masks[1:6]:
        # for i in masks:
        # for i in stable_masks:
            mask_image = i['segmentation'].astype(np.uint8) * 255
            resultimage = cv2.bitwise_and(image, image, mask=mask_image)

            # # Inverting B2W
            # black_pixels = np.where(
            #     (resultimage[:, :, 0] == 0) & 
            #     (resultimage[:, :, 1] == 0) & 
            #     (resultimage[:, :, 2] == 0)
            # )
            # resultimage[black_pixels] = [255,255,255]
            # #

            box = get_bounding_box(i, resultimage)
            big_boxes.append(box)
            cord = get_coordinates(i)
            cords.append(cord)
        big_boxes2 = [get_bounding_box(mask) for mask in masks[1:6]]

        # WHITE PADDING
        # cropped_output1 = np.array([B2W(tf.image.resize_with_crop_or_pad(box, 224, 224)) for box in big_boxes]) 
        # cropped_output2 = np.array([B2W(tf.image.resize_with_crop_or_pad(box, 224, 224)) for box in big_boxes2])

        # # BLACK PADDING
        cropped_output1 = np.array([tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes]) 
        cropped_output2 = np.array([tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes2])
        return cropped_output1, cropped_output2, cords, data

    else:
        cropped_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in bounding_boxes])

        np.savez_compressed('cropped_output', cropped_output)
    return cropped_output

    if(show_bboxes_grid=='y'):
        num_imgs = len(cropped_output)
        ncols = math.isqrt(num_imgs)
        nrows = -(num_imgs // (-ncols))
        fig = plt.figure(figsize=(20,20))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(nrows, ncols),
            axes_pad=0.1
        )
        for ax, im in zip(grid, cropped_output): ax.imshow(im)
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', '--device', choices=['none', 'mps', 'cuda'], 
                        default='cuda', help='Either none, mps, cuda')
    parser.add_argument('-so', '--show_original', choices=['y', 'n'], 
                        default='n',help='Show original image, [y] or n')
    parser.add_argument('-d', '--data', 
                        default='images/food_tray.jpg', help='Path to image to segment')
    parser.add_argument('-f', '--filter_method', choices=['none', 'stable', 'big'], 
                        default='big',help='Filtermethod for bounding box outputs')
    parser.add_argument('-sb', '--show_bboxes_grid', choices=['y','n'], 
                        default='n', help='Show bounding boxes, [y] or n')
    parser.add_argument('-sm', '--show_masks', choices=['y','n'], 
                        default='n', help='Show masks, [y] or n')
    parser.add_argument('-se', '--show_each', choices=['y','n'],
                        default='n',help='Show each individual bounding box')
    args = parser.parse_args()
    print('args:', args)

    run_model(device=args.device, 
              show_original=args.show_original, 
              data=args.data, 
              filter_method=args.filter_method, 
              show_bboxes_grid=args.show_bboxes_grid,
              show_masks=args.show_masks,
              show_each=args.show_each)

# python3 sam_auto_tutorial.py -d 'images/food_tray.jpg' -dev none -f none -sb y -sm y
