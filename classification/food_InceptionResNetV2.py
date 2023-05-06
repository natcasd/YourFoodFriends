import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
from keras.layers import \
       Conv2D, MaxPool2D, MaxPooling2D, Dropout, Flatten, Dense, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Add, \
            ZeroPadding2D, AveragePooling2D
import argparse
import pickle
from keras import mixed_precision
import hyperparameters as hp
import os
from datetime import datetime



num_classes = 101
# Black magic wtf
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
mixed_precision.global_policy()

def create_model():
    # Create base model
    input_shape = (299, 299, 3)
    base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False # freeze base model layers

    prediction = layers.Dense(num_classes, name='prediction')
    output = layers.Activation('softmax', dtype='float32', name='output')
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        # Dense(2048, activation='relu'),
        # Dropout(0.5),
        prediction,
        output
    ])

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
                optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
                metrics=["accuracy"])
    
    return model

# def run_model(load):
def run_model(load_checkpoint):
    (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                            split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                            shuffle_files=True, # shuffle files on download?
                                            as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                            with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

    def preprocess_img(image, label, img_shape=299):
        """
        Converts image datatype from 'uint8' -> 'float32' and reshapes image to
        [img_shape, img_shape, color_channels]
        """
        image = tf.image.resize(image, [299, 299]) # reshape to img_shape
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(image) #inception_resnet_v2 required
        return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

    # Map preprocessing function to training data (and paralellize)
    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle train_data and turn it into batches and prefetch it (load it faster)
    train_data = train_data.shuffle(hp.buffer_size).batch(hp.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map prepreprocessing function to test data
    test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Turn test data into batches (don't need to shuffle)
    test_data = test_data.batch(hp.batch_size).prefetch(tf.data.AUTOTUNE)

    '''
    # LOGGING
    '''
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    checkpoint_path = f"checkpoints/{timestamp}/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += 'weights.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="val_accuracy", # save the model weights with best validation accuracy
                                                        save_best_only=True, # only save the best weights
                                                        save_weights_only=False, # want entire model!
                                                        verbose=0) # don't print out whether or not model is being saved 
    
    log_path = f"logs/{timestamp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    model = create_model()
    if(load_checkpoint != 'n'):
        model.load_weights(load_checkpoint)
    summary_path = f"checkpoints/{timestamp}/summary.txt"
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.fit(
        train_data,
        epochs = 30,
        validation_data=test_data,
        callbacks = [checkpoint_callback, log_callback],
        verbose=2
    )
    '''
    # model_path = f"models/{timestamp}/model"
    # if not os.path.exists(model_path):
        # os.makedirs(model_path)
    # model.save(model_path)
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--load_checkpoint',
                        default='n',help='Load existing test/train data')
    
    args = parser.parse_args()
    print('args:', args)

    # run_model(load=args.load)
    run_model(load_checkpoint=args.load_checkpoint)