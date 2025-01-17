import tensorflow as tf
import tensorflow_datasets as tfds
from keras import mixed_precision
import argparse
import os
from datetime import datetime
import model_runners

num_classes = 101

# Accelerate on Nvidia
mixed_precision.set_global_policy(policy="mixed_float16") 
mixed_precision.global_policy()

def preprocess_image(image, label, img_shape=299):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    return tf.cast(image, tf.float32), label

def process_callbacks(loaded_model):
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    # CHECKPOINT CALLBACK
    checkpoint_path = f"../finetune_checkpoints/{timestamp}/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += 'model.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="val_accuracy", # Trigger on value accuracy
                                                        save_best_only=True, # Only save best monitor (val_accuracy)
                                                        save_weights_only=False, # Save entire model;
                                                        # save_weights_only=True, # Save only weights; USE WITH EFFICIENTNET
                                                        verbose=0) # Don't print save messages
    # TENSORBOARD LOGGING CALLBACK
    log_path = f"../logs/{timestamp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    # MODEL SUMMARY LOGGING
    summary_path = f"../finetune_checkpoints/{timestamp}/summary.txt"
    with open(summary_path, 'w') as f:
        loaded_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # CSV LOGGING CALLBACK
    csv_logger_path = f"../finetune_checkpoints/{timestamp}/epochs.csv"
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

    # MODEL FITTING CALLACKS
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                      patience=3) # Stop if val_accuracy doesn't improve after 3 epochs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",  
                                                     factor=0.2, # Reduce LR factor of 0.2
                                                     patience=2, # Trigger LR reduction if val_accuracy doesn't improve after 2 epochs
                                                     verbose=1,
                                                     min_lr=1e-7)
    
    return [checkpoint_callback, log_callback, csv_logger_callback, early_stopping, reduce_lr]

def run_model(load_checkpoint):
    # Load and process data
    (train_data, test_data), ds_info = tfds.load(name="food101", 
                                            split=["train", "validation"], 
                                            shuffle_files=True,
                                            as_supervised=True,
                                            with_info=True)
    
    # Preprocess training and test data
    train_data = train_data.map(map_func=preprocess_image, 
                                num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.map(preprocess_image, 
                              num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch data
    train_data = train_data.shuffle(32).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE) #Shuffle!
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE) 

    # Create the model
    loaded_model = model_runners.food_IRNv2.create_model()
    loaded_model.load_weights('../checkpoints/050523-180311/weights.hdf5') #050523-180311 best IRNv2

    # Unfreeze all layers
    for layer in loaded_model.layers:
        layer.trainable = True
    
    # Compile the model
    loaded_model.compile(loss="sparse_categorical_crossentropy",
                            optimizer=tf.keras.optimizers.Adam(0.0001),
                            metrics=["accuracy"])
    
    # Load callbacks
    processed_callbacks = process_callbacks(loaded_model)

    # Fit the model
    loaded_model.fit(
        train_data,
        epochs=100,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=int(0.15 * len(test_data)), # Don't have to validate everything
        callbacks = processed_callbacks,
        verbose=2
    )

    '''
    # This is an alternative to model saving
    unfrozen_model_path = f'../finetune_checkpoints/{timestamp}/unfrozen_model'
    if not os.path.exists(unfrozen_model_path):
        os.makedirs(unfrozen_model_path)
    
    def un_freeze(loaded_model):
        for layer in loaded_model.layers:
            layer.trainable = True
        
            if isinstance(layer, models.Model):
                un_freeze(layer)
        loaded_model.save(unfrozen_model_path)

    un_freeze(loaded_model)

    frozen_model_path = f'../finetune_checkpoints/{timestamp}/frozen_model'
    if not os.path.exists(frozen_model_path):
        os.makedirs(frozen_model_path)
    
    def freeze(loaded_model):
        for layer in loaded_model.layers:
            layer.trainable = False
        
            if isinstance(layer, models.Model):
                freeze(layer)
        loaded_model.save(frozen_model_path)
    freeze(loaded_model)
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--load_checkpoint',
                        default='n',help='Load existing test/train data')
    
    args = parser.parse_args()
    print('args:', args)

    run_model(load_checkpoint=args.load_checkpoint)