import numpy as np
from segmentation import sam_auto_tutorial
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
img_shape = 224

counter = 0
def classify_list(images):
    global counter
    with open('classification/class_names.pkl', 'rb') as f:
        class_list = pickle.load(f)[0]
    
    # Load Model Strategy
    model = tf.keras.models.load_model('classification/finetune_checkpoints/050823-171401/model.hdf5')
    
    # Instantiate accumulator lists for classifier predictions
    label_list = []
    max_list = []

    # Predict each image
    for i in images:
        '''
        # Save masks
        plt.imshow(i)
        plt.savefig(f'fnf/experiments/mask{counter}.jpg') # Save masks
        '''

        image = preprocess_img(i)
        image = tf.expand_dims(image,0)
        predictions = model.predict(image, verbose=0)
        labelarg = np.argmax(predictions)
        max = np.max(predictions)
        max_list.append(max)
        label_list.append(class_list[labelarg])
        counter += 1

    return label_list, max_list

def preprocess_img(image):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [299, 299]) # reshape to img_shape # Already resized
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image) #vgg16 required
    return tf.cast(image, tf.float32) # return (float32_image, label) tuple

# Obtain SAM masks
segments1, segments2, coordinatelist, imageloc = \
    sam_auto_tutorial.run_model(
    device='cuda', 
    filter_method='boundingboxes', 
    data = 'segmentation/images/team_picture.png')

# Obtain classifier predictions
labeled_list1, max_list1 = classify_list(segments1) # Pdns on cropped BBs
labeled_list2, max_list2 = classify_list(segments2) # Pdns on uncropped BBs

# Backbone original image
wholeimage = plt.imread(imageloc)
fig, ax = plt.subplots(1)
ax.imshow(wholeimage)

# For each classifier prediction, draw bounding box w/ label
for i,j in enumerate(coordinatelist):

    # Ensemble vote between cropped and un-cropped bounding boxes
    if max(max_list1[i], max_list2[i]) > 0.45:
        # Draw rectangle about bounding box
        rect = patches.Rectangle(
            (j[0], j[1]), j[2], j[3],
            linewidth=2,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)

        # Label positioning
        labelx = j[0]
        labely = j[1]-5 # Slightly above the top-left corner

        # Add better label
        if(max_list1[i]>=max_list2[i]):
            ax.text(labelx, labely, labeled_list1[i], fontsize=12, color='k', weight='bold')
        else:
            ax.text(labelx, labely, labeled_list2[i], fontsize=12, color='k', weight='bold')

# plt.show()
plt.savefig('output.png')