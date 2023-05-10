import numpy as np
from segmentation import sam_auto_tutorial
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import cv2

counter = 0
def classify_list(images):
    global counter
    class_list = ['food', 'not_food']
        
    fnf_model = tf.keras.models.load_model('checkpoints/050823-195400/models.hdf5')
    label_list = []
    max_list = []
    
    for i in images:
        image = tf.image.resize(i, [256, 256])
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image,0)
        predictions = fnf_model.predict(image, verbose=0)
        labelarg = np.argmax(predictions)
        max = np.max(predictions)
        pred_label = class_list[labelarg]
        max_list.append(max)
        label_list.append(pred_label)
        plt.title('fnf predictions: ' + pred_label)
        plt.imshow(i)
        save_path = f'experiments/{pred_label}/{counter}.jpg'
        plt.savefig(save_path)
        counter += 1

    print('exiting classifyList')
    return label_list, max_list

segments1, segments2, coordinatelist, imageloc = sam_auto_tutorial.run_model(device='cuda', filter_method='boundingboxes', data = 'segmentation/images/ratty1.jpg')
labeled_list1, max_list1 = classify_list(segments1)
labeled_list2, max_list2 = classify_list(segments2)