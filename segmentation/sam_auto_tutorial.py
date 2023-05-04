import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import tensorflow as tf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# cutouts = []
# def run_model(device='none', show_original='n', data='images/dog.jpg', filter_method='big', show_bboxes='n', show_masks='n'):
def run_model(device, show_original, data, filter_method, show_bboxes, show_masks):
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

    if(show_original=='y'):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    
    sys.path.append("..")
    

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # device = torch.device("mps") # Apple M# Metal
    # device = 'cuda' # CUDA
    # if no device, set to 0
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    if(device == 'cuda'):
        device = 'cuda'
        sam.to(device=device)
    elif(device == 'mps'):
        device = torch.device('mps')
        sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

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
    def get_bounding_box(mask):
        XYWH = mask['bbox']
        return image[XYWH[1]:XYWH[1]+XYWH[3], XYWH[0]:XYWH[0]+XYWH[2]]
    
    for mask in masks:
        prediction = mask['predicted_iou']
        stability = mask['stability_score']
        area = mask['area']

        bounding_box = get_bounding_box(mask)
        if(show_bboxes=='y'):
            plt.imshow(bounding_box)
            plt.title(f'Confidence:{prediction}\nStability:{stability}\nArea:{area}')
            plt.show() #TODO: White out everything except mask
        bounding_boxes.append(bounding_box)
    

    if(filter_method=='stable'):
        stability_filter = 0.97
        filtered_stable_masks = filter(lambda mask : mask['stability_score'] > stability_filter, masks)
        cropped_stable_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in stable_boxes])
        stable_boxes = [get_bounding_box(mask) for mask in filtered_stable_masks]
        np.savez_compressed('cropped_stable_output', cropped_stable_output)

            

    elif(filter_method=='big'):
        big_filter = 30000
        filtered_big_masks = filter(lambda mask: mask['area'] > big_filter, masks)
        big_boxes = [get_bounding_box(mask) for mask in filtered_big_masks]

        cropped_big_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in big_boxes])
        np.savez_compressed('cropped_big_output', cropped_big_output)

    else:
        cropped_output = np.array([
            tf.image.resize_with_crop_or_pad(box, 224, 224) for box in bounding_boxes])
        np.savez_compressed('cropped_output', cropped_output)


    
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', '--device', choices=['none', 'mps', 'cuda'], 
                        default='none', help='Either none, mps, cuda')
    parser.add_argument('-so', '--show_original', choices=['y', 'n'], 
                        default='n',help='Show original image, [y] or n')
    parser.add_argument('-d', '--data', 
                        default='images/food_tray.jpg', help='Path to image to segment')
    parser.add_argument('-f', '--filter_method', choices=['none', 'stable', 'big'], 
                        default='big',help='Filtermethod for bounding box outputs')
    parser.add_argument('-sb', '--show_bboxes', choices=['y','n'], 
                        default='n', help='Show bounding boxes, [y] or n')
    parser.add_argument('-sm', '--show_masks', choices=['y','n'], 
                        default='n', help='Show masks, [y] or n')
    args = parser.parse_args()
    print('args:', args)

    run_model(device=args.device, 
              show_original=args.show_original, 
              data=args.data, 
              filter_method=args.filter_method, 
              show_bboxes=args.show_bboxes,
              show_masks=args.show_masks)

