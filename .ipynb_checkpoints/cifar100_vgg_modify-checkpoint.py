"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard
import os
import tensorflow as tf
import argparse

import numpy as np
import vgg7
import resnet
#args 
parse = argparse.ArgumentParser()
parse.add_argument("--layer_nums",help="the num of layers in each branch",default=18,type=int)
parse.add_argument("--input_layers",help="the num of layers in each branch",nargs=4,default=[2,2,2,2],type=int)
parse.add_argument("--backbone",help="backbone of the network",default="vgg7",type=str)
parse.add_argument("--logPath",help="the path for log ",default="./logs/lalala/",type=str)
parse.add_argument("--only_last",help="nouse merge use only last branch ",default=-1,type=int)
args = parse.parse_args()


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.00000000001, patience=100)
csv_logger = CSVLogger('vgg7_modify_cifar100.csv')

batch_size = 32
nb_classes = 100
nb_epoch = 400
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR100 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.
#create model
if(args.layer_nums==18):
    model = vgg7.vggBuilder.build_three_branch_18((img_channels, img_rows, img_cols), nb_classes)
elif(args.layer_nums==34):
    model = vgg7.vggBuilder.build_three_branch_32((img_channels, img_rows, img_cols), nb_classes)
elif(args.layer_nums==50):
    model = vgg7.vggBuilder.build_three_branch_50((img_channels, img_rows, img_cols), nb_classes)
elif(args.layer_nums==101):
    model = vgg7.vggBuilder.build_three_branch_101((img_channels, img_rows, img_cols), nb_classes)
elif(args.layer_nums==152):
    model = vgg7.vggBuilder.build_three_branch_152((img_channels, img_rows, img_cols), nb_classes)
if(args.only_last==0):
    model = vgg7.vggBuilder.build_three_branch_18_only_last((img_channels, img_rows, img_cols), nb_classes,args.only_last)
elif(args.only_last==1):
    model = vgg7.vggBuilder.build_three_branch_18_only_last_model1((img_channels, img_rows, img_cols), nb_classes,args.only_last)
elif(args.only_last==2):
    model = vgg7.vggBuilder.build_three_branch_18_only_last_model2((img_channels, img_rows, img_cols), nb_classes,args.only_last)   
if(args.input_layers!=None):
    model = vgg7.vggBuilder.build_three_branch_input((img_channels, img_rows, img_cols), nb_classes,args.input_layers)

if(args.backbone=="vgg7"):
    model = vgg7.vggBuilder.vgg7((img_channels, img_rows, img_cols), nb_classes)
elif(args.backbone=="vggA"):
    model = vgg7.vggBuilder.vggA((img_channels, img_rows, img_cols), nb_classes) 
elif(args.backbone=="vggB"):
    model = vgg7.vggBuilder.vggB((img_channels, img_rows, img_cols), nb_classes)
elif(args.backbone=="vggC"):
    model = vgg7.vggBuilder.vggC((img_channels, img_rows, img_cols), nb_classes)
elif(args.backbone=="vggD"):
    model = vgg7.vggBuilder.vggD((img_channels, img_rows, img_cols), nb_classes)
elif(args.backbone=="resnet18"):
    model = resnet.ResnetBuilder.build_resnet_18_old((img_channels, img_rows, img_cols), nb_classes)

    
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

logPath = args.logPath
if(not os.path.exists(logPath)):
    os.makedirs(logPath)
    
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger,TensorBoard(log_dir=logPath)])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[TensorBoard(log_dir=logPath),lr_reducer, early_stopper, csv_logger])
