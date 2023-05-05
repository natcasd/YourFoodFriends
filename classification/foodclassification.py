import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import argparse
import sys
import pickle

num_classes = 101

def create_model():
    # Create base model
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False # freeze base model layers

    flatten = layers.Flatten()
    dense1 = layers.Dense(1000, activation='relu')
    dense2 = layers.Dense(500, activation='relu')
    prediction = layers.Dense(num_classes, activation='softmax')

    #outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
    model = models.Sequential([
            base_model,
            flatten,
            dense1,
            dense2,
            prediction
        ])

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["accuracy"])
    
    return model

def run_model(load):
    if(load=='n'):
        (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                                    split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                                    shuffle_files=True, # shuffle files on download?
                                                    as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                                    with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

        class_names = ds_info.features["label"].names

        with open('data.pkl', 'wb') as f:
            pickle.dump([train_data, test_data, ds_info, class_names], f)
    elif(load=='y'):
        with open('data.pkl', 'rb') as f:
            train_data, test_data, ds_info, class_names = pickle.load(f)
    else:
        (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                                    split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                                    shuffle_files=True, # shuffle files on download?
                                                    as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                                    with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

        class_names = ds_info.features["label"].names


    def preprocess_img(image, label, img_shape=224):
        """
        Converts image datatype from 'uint8' -> 'float32' and reshapes image to
        [img_shape, img_shape, color_channels]
        """
        image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
        image = tf.keras.applications.vgg16.preprocess_input(image) #vgg16 required
        return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

    # Map preprocessing function to training data (and paralellize)
    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle train_data and turn it into batches and prefetch it (load it faster)
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map prepreprocessing function to test data
    test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Turn test data into batches (don't need to shuffle)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    checkpoint_path = "model_checkpoints/cp.ckpt" # saving weights requires ".ckpt" extension
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        monitor="val_accuracy", # save the model weights with best validation accuracy
                                                        save_best_only=True, # only save the best weights
                                                        save_weights_only=True, # only save model weights (not whole model)
                                                        verbose=0) # don't print out whether or not model is being saved 

    model = create_model()
    model.fit(
        train_data,
        epochs=3,
        validation_data=test_data,
        callbacks = [model_checkpoint]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--load', choices=['y','n','q'],
                        default='n',help='Load existing test/train data')
    
    args = parser.parse_args()
    print('args:', args)

    run_model(load=args.load)