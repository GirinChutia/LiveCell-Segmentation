import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Concatenate,Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform,HeUniform
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import numpy as np
import segmentation_models as sm
from segmentation_models.metrics import iou_score
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import albumentations as A

K.set_image_data_format('channels_last')
sm.set_framework('tf.keras')

class identity_block(tf.keras.layers.Layer):
    def __init__(self, kernel=3,  filters=[4,4,8], name="identity_block"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters
        
        self.kernel = kernel
        
        self.conv_1x1_1 = Conv2D(self.F1,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer=HeUniform)
        
        self.bn1 = BatchNormalization(axis=-1)
        
        self.conv_3x3 = Conv2D(self.F2,
                             kernel_size=(self.kernel,self.kernel),
                             strides=(1,1),
                             padding='same',
                             kernel_initializer=HeUniform)
        
        self.bn2 = BatchNormalization(axis=-1)
        
        self.conv_1x1_2 = Conv2D(self.F3,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer=HeUniform,)
        
        self.bn3 = BatchNormalization(axis=-1)
        self.activation = Activation('relu')
        self.add = Add()
        
    def call(self, X):
        # write the architecutre that was mentioned above
        X_input = X
        
        X = self.conv_1x1_1(X)
        X = self.bn1(X)
        X = self.activation(X)
        
        X = self.conv_3x3(X)
        X = self.bn2(X)
        X = self.activation(X)
        
        X = self.conv_1x1_2(X)
        X = self.bn3(X)
        X = self.activation(X)
        
        X = self.add([X,X_input])
        X = self.activation(X)
        return X
    
    def get_config(self):
        config = super().get_config()
        config.update({"kernel": self.kernel})
        return config
    
class convolutional_block(tf.keras.layers.Layer):
    def __init__(self, kernel=3,  filters=[4,4,8], stride=2, name="conv_block"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters #no of filters
        self.kernel = kernel #filter size
        self.stride = stride #stride
        
        self.conv_1 = Conv2D(self.F1,
                             kernel_size=(1,1),
                             strides=(1,1),
                             padding='same',
                             kernel_initializer=HeUniform)
        
        self.bn1 = BatchNormalization(axis=-1) #axis = -1 for channel last, 1 for channel first
        
        self.conv_2 = Conv2D(self.F2,
                             kernel_size=(self.kernel,self.kernel),
                             strides=(self.stride,self.stride),
                             padding='same',
                             kernel_initializer=HeUniform)
        
        self.bn2 = BatchNormalization(axis=-1)
        
        self.conv_3 = Conv2D(self.F3,
                             kernel_size=(1,1),
                             strides=(1,1),
                             padding='same',
                             kernel_initializer=HeUniform)
        
        self.bn3 = BatchNormalization(axis=-1)
         
        self.conv_parallel =  Conv2D(self.F3,
                                     kernel_size=(self.kernel,self.kernel),
                                     strides=(self.stride,self.stride),
                                     padding='same',
                                     kernel_initializer=HeUniform)
        
        self.bn_parallel = BatchNormalization(axis=-1)
        
        self.add = Add()
        self.activation = Activation('relu')
        
    def call(self, X):
    
        X_input = X
        # First Convolutional Block
        conv1x1_1 = self.conv_1(X)
        bn1 = self.bn1(conv1x1_1)
        act1 = self.activation(bn1)
        
        # Second Convolutional Block
        conv3x3 = self.conv_2(act1)
        bn2 = self.bn2(conv3x3)
        act2 = self.activation(bn2)
        
        # Third Convolutional Block
        conv1x1_2 = self.conv_3(act2)
        bn3 = self.bn3(conv1x1_2)
        
        #Parallel block
        conv_parallel = self.conv_parallel(X_input)
        bnp = self.bn_parallel(conv_parallel)
        actp = self.activation(bnp)
        
        # add the input
        X = self.add([bn3, actp])
        X = self.activation(X)
                            
        return X
    
    def get_config(self):
        config = super().get_config()
        config.update({"kernel": self.kernel,
                       "stride": self.stride,
                       "F1" : self.F1, 
                       "F2" : self.F2,
                       "F3": self.F3,"conv1": self.conv_1,
                       "conv2":self.conv_2,
                       "conv3":self.conv_3})
        return config

class global_flow(tf.keras.layers.Layer):
    def __init__(self,up_w,up_h,do_upsample=False,channels=64,name="global_flow"):
        super().__init__(name=name)
        self.do_upsample = do_upsample
        self.up_w = up_w
        self.up_h = up_h
        self.glob_avg_pool = GlobalAveragePooling2D(data_format='channels_last',keepdims=True)
        self.conv_1x1 = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=HeUniform) 
        self.bn = BatchNormalization(axis=-1)
        self.activation = Activation("relu")
        self.upsample = UpSampling2D(size=(self.up_w,self.up_h),
                                     interpolation='bilinear',
                                     data_format='channels_last')
        
    def call(self, X):
        X = self.glob_avg_pool(X)
        X = self.conv_1x1(X) 
        X = self.bn(X)
        X = self.activation(X)
        if self.do_upsample ==True:
            X = self.upsample(X)
        return X

class context_flow(tf.keras.layers.Layer):
        
    def __init__(self, name="context_flow",channels=64,N=2):
        super().__init__(name=name)
        self.concat_globalflow_c4 = Concatenate(axis=-1)
        
        if N == 2:
            ps = (2,2)
        if N == 4:
            ps = (4,4)
            
        self.avg_pool = AveragePooling2D(pool_size=ps)
        self.upsample = UpSampling2D(size=ps,interpolation='bilinear')
        
        self.conv1 = Conv2D(channels,kernel_size=(3,3),padding='same',kernel_initializer=HeUniform,activation='relu')
        self.conv2 = Conv2D(channels,kernel_size=(3,3),padding='same',kernel_initializer=HeUniform,activation='relu')
        
        self.conv_1x1_1 = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=HeUniform) 
        self.conv_1x1_2 = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=HeUniform) 

        self.relu = Activation('relu')
        self.sigmoid = Activation("sigmoid")
        
        self.add = Add()
        self.multiply = Multiply()
        
    def call(self, X):
        # here X will a list of two elements 
        INP, FLOW = X[0], X[1]
        
        concat = self.concat_globalflow_c4([INP, FLOW])
        avgpool = self.avg_pool(concat)
        
        conv1 = self.conv1(avgpool)
        conv2 = self.conv2(conv1)
        
        conv1x1_1 = self.conv_1x1_1(conv2)
        relu = self.relu(conv1x1_1)
        
        conv1x1_2 = self.conv_1x1_2(relu)
        sigmoid = self.sigmoid(conv1x1_2)
        
        _mul = self.multiply([conv2,sigmoid])
        _add = self.add([conv2,_mul])
        X = self.upsample(_add)
        
        return X

class fsm(tf.keras.layers.Layer):    
    def __init__(self, name="feature_selection",channels=64):
        super().__init__(name=name)
        
        self.conv = Conv2D(filters=channels,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding='same',
                           kernel_initializer=HeUniform,
                           activation='relu')
        
        self.glob_avg_pool = GlobalAveragePooling2D(data_format='channels_last',keepdims=True)
        
        self.conv_1x1 = Conv2D(channels,kernel_size=(1,1),
                               strides=(1,1),padding='same',
                               kernel_initializer=HeUniform)
        
        self.bn = BatchNormalization(axis=-1)
        self.sigmoid = Activation("sigmoid") 
        self.multiply = Multiply()
        self.upsample = UpSampling2D(size=(2,2),interpolation='bilinear')
        
    def call(self, X):
        X_conv = self.conv(X)
        X = self.glob_avg_pool(X_conv)
        X = self.conv_1x1(X)
        X = self.bn(X)
        X = self.sigmoid(X)
        X = self.multiply([X_conv,X])
        FSM_Conv_T = self.upsample(X)
        return FSM_Conv_T

class agcn(tf.keras.layers.Layer):    
    def __init__(self, name="global_conv_net",channels=32):
        super().__init__(name=name)
        
        self.conv_17_1 = Conv2D(filters=channels,kernel_size=(1,7),padding='same',kernel_initializer=HeUniform,activation='relu')
        self.conv_71_1 = Conv2D(filters=channels,kernel_size=(7,1),padding='same',kernel_initializer=HeUniform,activation='relu')
        
        self.conv_17_2 = Conv2D(filters=channels,kernel_size=(1,7),padding='same',kernel_initializer=HeUniform,activation='relu')
        self.conv_71_2 = Conv2D(filters=channels,kernel_size=(7,1),padding='same',kernel_initializer=HeUniform,activation='relu')
        
        self.conv_3x3 = Conv2D(filters=channels,kernel_size=(3,3),padding='same',kernel_initializer=HeUniform,activation='relu')
        self.add = Add()
        
    def call(self, X):
        x_p1 = self.conv_71_1(X)
        x_p1 = self.conv_17_1(x_p1)
        
        x_p2 = self.conv_17_2(X)
        x_p2 = self.conv_71_2(x_p2)
        
        x_add = self.add([x_p1,x_p2])
        
        x_3x3 = self.conv_3x3(x_add)
        
        y = self.add([x_3x3,x_add])
        
        return y