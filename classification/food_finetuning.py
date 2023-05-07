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
# import hyperparameters as hp
import os
from datetime import datetime
# from food_ENB3 import create_model
# from food_InceptionResNetV2 import create_model
from food_MNv2 import create_model



num_classes = 101
# Black magic wtf
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
mixed_precision.global_policy()

def run_model(load_checkpoint):
    #########################
    # LOAD AND PROCESS DATA
    #########################
    (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                            split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                            shuffle_files=True, # shuffle files on download?
                                            as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                            with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)
    
    def preprocess_img(image, label, img_shape=224):
        """
        Converts image datatype from 'uint8' -> 'float32' and reshapes image to
        [img_shape, img_shape, color_channels]
        """
        image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #not required
        return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

    # Map preprocessing function to training data (and paralellize)
    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle train_data and turn it into batches and prefetch it (load it faster)
    train_data = train_data.shuffle(32).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map prepreprocessing function to test data
    test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Turn test data into batches (don't need to shuffle)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    #########################
    # CREATE MODEL
    #########################
    loaded_model = create_model()
    loaded_model.load_weights('checkpoints/050723-133413/weights.hdf5')

    for layer in loaded_model.layers:
        layer.trainable = True
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
    
    # Compile the model
    loaded_model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                            optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                            metrics=["accuracy"])
    
    #########################
    # CALLBACKS
    #########################

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    # CHECKPOINT CALLBACK
    checkpoint_path = f"finetune_checkpoints/{timestamp}/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += 'weights.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="val_accuracy", # save the model weights with best validation accuracy
                                                        save_best_only=True, # only save the best weights
                                                        save_weights_only=True, # want entire model!
                                                        verbose=0) # don't print out whether or not model is being saved 
    # TENSORBOARD LOGGING CALLBACK
    log_path = f"logs/{timestamp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    # MODEL SUMMARY LOGGING
    summary_path = f"finetune_checkpoints/{timestamp}/summary.txt"
    with open(summary_path, 'w') as f:
        loaded_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # CSV LOGGING CALLBACK
    csv_logger_path = f"finetune_checkpoints/{timestamp}/epochs.csv"
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

    # MODEL FITTING CALLACKS
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", # watch the val loss metric
                                                    patience=3) # if val loss decreases for 3 epochs in a row, stop training
    # Creating learning rate reduction callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",  
                                                    factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                    patience=2,
                                                    verbose=1, # print out when learning rate goes down 
                                                    min_lr=1e-7)

    #########################
    # FIT MODEL
    #########################
    loaded_model.fit(
        train_data,
        epochs = 100,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=int(0.15 * len(test_data)),
        callbacks = [checkpoint_callback, log_callback, csv_logger_callback, early_stopping, reduce_lr],
        verbose=2
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--load_checkpoint',
                        default='n',help='Load existing test/train data')
    
    args = parser.parse_args()
    print('args:', args)

    # run_model(load=args.load)
    run_model(load_checkpoint=args.load_checkpoint)