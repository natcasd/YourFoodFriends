'''
This file contains a sequence of models for VGG Head
'''
import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, MaxPooling2D, Dropout, Flatten, Dense, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Add, \
            ZeroPadding2D, AveragePooling2D
from keras.layers import Input

import hyperparameters as hp
# VGG Head
leaky_gap_d05bet_1024_512_256= [
            GlobalAveragePooling2D(),
            Dense(1024, activation='leaky_relu'),
            Dropout(0.5),
            Dense(512, activation='leaky_relu'),
            Dropout(0.5),
            Dense(256, activation='leaky_relu'),
            Dropout(0.5),
        ]
leaky_gap_d05bet_4096_2048_1024= [
            GlobalAveragePooling2D(),
            Dense(4096, activation='leaky_relu'),
            Dropout(0.5),
            Dense(2048, activation='leaky_relu'),
            Dropout(0.5),
            Dense(1024, activation='leaky_relu'),
            Dropout(0.5)
        ]
gap_d05bet_1024_512_256= [
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5)
        ]
gap_d05bet_4096_2048_1024= [
            GlobalAveragePooling2D(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(2048, activation='relu'),
            Dropout(0.5),
            Dense(1024, activation='relu'),
            Dropout(0.5)        
        ]

gap_d05bet_2048_2048= [
            GlobalAveragePooling2D(),
            Dense(2048, activation='relu'),
            Dropout(0.5),
            Dense(2048, activation='relu'),
            Dropout(0.5)
        ]