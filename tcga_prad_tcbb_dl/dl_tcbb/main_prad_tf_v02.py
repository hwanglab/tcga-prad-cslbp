# reference:https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # use gpu 1

import time
import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import argparse
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras import optimizers
from keras import applications
from keras.models import Model


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

train_dir = '../../data_patches/fold3/train/'
valid_dir = '../../data_patches/fold3/valid/'
batch_size = 64
img_width, img_height = 224, 224
epochs = 100

tensorboard_path = '../../models/tf_v2/tensorboard/'
models_path = '../../models/tf_v2/fold03/'


#kk=3
#ps=2

def cal_train_validation_size():
    train_num = 0
    valid_num = 0

    # train images
    subclass = os.listdir(train_dir)
    for i in range(len(subclass)):
        temp_path = train_dir + subclass[i] + '/'
        train_num += len(glob.glob(temp_path + '*.png'))

    # valid image
    subclass = os.listdir(valid_dir)
    for i in range(len(subclass)):
        temp_path = valid_dir + subclass[i] + '/'
        valid_num += len(glob.glob(temp_path + '*.png'))

    return train_num, valid_num


def build_transfer_learning_model():
    model_base = applications.VGG16(weights='imagenet',include_top=False,input_shape=(img_width,img_height,3))

    layer_num=len(model_base.layers)-3
    for layer in model_base.layers[:layer_num]:
        layer.trainable=False

    model_base.summary()

    #v2: tune the last 4 layers + output layer
    # add only output layer
    x = model_base.output
    x = Conv2D(64, (1,1), activation='relu',padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x=Dropout(0.5)(x)
    pred=Dense(3, activation='softmax')(x)

    model = Model(input=model_base.input, output=pred)

    #v1: frozen all vgg16 layers but add a few below layers
    # bottleneck_shape=model_base.layers[-1].output_shape #16x16x512 for 512x512 inputs
    # #adding custer layers
    # x=model_base.output
    # x=Conv2D(512, (1, 1), activation='relu', padding='same')(x)
    # x=Conv2D(512, (1, 1), activation='relu', padding='same')(x)
    # x=Conv2D(512, (1, 1), activation='relu', padding='same')(x)
    # x=GlobalAveragePooling2D()(x)
    #
    # #model.add(Flatten())
    # x=Dense(512, activation='relu')(x)
    # x=Dropout(0.5)(x)
    # pred=Dense(2, activation='softmax')(x)
    #
    # model=Model(input=model_base.input,output=pred)

    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])
    return model

def preprocessor(input):
    image_aug = image.img_to_array(input)
    for k in range(3):
        ac=np.random.uniform(0.9,1.1)
        bc=np.random.uniform(-5,5)
        image_aug[:,:,k]=ac*image_aug[:,:,k]+bc

    image_aug = np.clip(image_aug, 0, 255)
    image_aug=image.array_to_img(image_aug)
    return image_aug


def train_model():
    train_num, valid_num = cal_train_validation_size()

    # check to see if we are compiling using just a single GPU
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = build_transfer_learning_model()

    # otherwise, we are compiling using multiple GPUs
    else:
         print("[INFO] training with {} GPUs...".format(G))

         # we'll store a copy of the model on *every* GPU and then combine
         # the results from the gradient updates on the CPU
         with tf.device("/cpu:0"):
             # initialize the model
             model = build_transfer_learning_model()

         # make the model parallel
         model = multi_gpu_model(model, gpus=G)

    print(model.summary())


    train_datagen = ImageDataGenerator(
        # transformation first
        rotation_range=5,
        zoom_range=0.05,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=10,
        brightness_range=[0.9, 1.1],

        # standardization second
        preprocessing_function=preprocessor,
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      samplewise_center=True,
                                      samplewise_std_normalization=True)

    # Training new model
    ts = str(int(time.time()))
    model_name = 'VGG16'
    run_name = 'model={}-batch_size={}-ts={}'.format(model_name, batch_size, ts)
    tensorboard_loc = os.path.join(tensorboard_path, run_name)
    checkpoint_loc = os.path.join(models_path, 'model-{}-weights.h5'.format(ts))

    earlyStopping = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                  patience=10,          # number of epochs with no improvement after which training will be stopped
                                  verbose=1,           # decides what to print
                                  min_delta=0.0001,    # minimum change to qualify as an improvement
                                  mode='min', )        # min mode: training will stop when the quantity monitored has stopped decreasing

    modelCheckpoint = ModelCheckpoint(checkpoint_loc,       # path to save the model, save the model after every epoch
                                      monitor='val_loss',   # quantity to monitor
                                      save_best_only=True,  # if ture, the latest best model will not be overwritten
                                      mode='min',           # the decision to overwrite based on
                                      verbose=1,
                                      save_weights_only=True)

    tensorboard = TensorBoard(log_dir=tensorboard_loc, histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [modelCheckpoint, earlyStopping, tensorboard]


    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model_json = model.to_json()
    with open(models_path + "model_vgg16_fold3.json", "w") as json_file:
        json_file.write(model_json)
        print("save model to disk")


    model.fit_generator(
        train_generator,
        steps_per_epoch=train_num // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=valid_num // batch_size,
        callbacks=callbacks_list)

    #model.save_weights('frist_try.h5')
    # save the trained model
    #model_json = model.to_json()
    #with open("model_first_try.json", "w") as json_file:
    #    json_file.write(model_json)
    #    print("save model to disk")


if __name__ == '__main__':
    train_model()
    print('training done!!')

# cd projects/tcga_prad/tcga_prad_tcbb_dl/dl_tcbb