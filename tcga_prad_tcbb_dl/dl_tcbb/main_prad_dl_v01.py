# this looks like the best we have

import os
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import argparse
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

fold=3

if fold==1:
    Train_data=['../fold1/train/g6/',
                '../fold1/train/g7/',
                '../fold1/train/g8_9']
    Validation_data=['../fold1/valid/g6/',
                     '../fold1/valid/g7/',
                     '../fold1/valid/g8_9']

    train_path='../fold1/train/'
    valid_path='../fold1/valid/'

    TENSORBOARD_PATH='./fold01_dl/tensorboard/'
    MODELS_PATH='./fold01_dl/models/'

elif fold==2:
    Train_data = ['../fold2/train/g6/',
                  '../fold2/train/g7/',
                  '../fold2/train/g8_9']
    Validation_data = ['../fold2/valid/g6/',
                       '../fold2/valid/g7/',
                       '../fold2/valid/g8_9']

    train_path = '../fold2/train/'
    valid_path = '../fold2/valid/'

    TENSORBOARD_PATH = './fold02_dl/tensorboard/'
    MODELS_PATH = './fold02_dl/models/'
elif fold==3:
    Train_data = ['../fold3/train/g6/',
                  '../fold3/train/g7/',
                  '../fold3/train/g8_9']
    Validation_data = ['../fold3/valid/g6/',
                       '../fold3/valid/g7/',
                       '../fold3/valid/g8_9']

    train_path = '../fold3/train/'
    valid_path = '../fold3/valid/'

    TENSORBOARD_PATH = './fold03_dl/tensorboard/'
    MODELS_PATH = './fold03_dl/models/'
else:
    print('not for this work testing!!!!!')

batch_size = 128
rn=256
cn=256
kk=3
ps=3

def get_img_id():
    train_ids = []
    for k in range(0, len(Train_data)):
        path_temp = Train_data[k]
        images = os.listdir(path_temp)
        for image_name in images:
            if '.png' or '.jpg' in image_name:
                train_ids.append(image_name)

    validation_ids = []
    for j in range(0, len(Validation_data)):
        path_temp = Validation_data[j]
        images = os.listdir(path_temp)
        for image_name in images:
            if '.png' or '.jpg' in image_name:
                validation_ids.append(image_name)

    return train_ids, validation_ids


def get_VGG_Like_model():
    model = Sequential()
    model.add(Conv2D(8, (kk, kk), input_shape=(rn, cn, 3),padding='same',activation='relu'))  # in input_shape: the batch dimension is not included
    model.add(Conv2D(8, (kk, kk), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(ps, ps)))

    model.add(Conv2D(16, (kk, kk), padding='same', activation='relu'))
    model.add(Conv2D(16, (kk, kk), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(ps, ps)))

    model.add(Conv2D(32, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(32, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(32, (kk, kk),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(ps, ps)))

    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(ps, ps)))

    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(Conv2D(64, (kk, kk),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(ps, ps)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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


def train_classification():
    train_ids, validation_ids = get_img_id()

    # check to see if we are compiling using just a single GPU
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = get_VGG_Like_model()

    # otherwise, we are compiling using multiple GPUs
    else:
         print("[INFO] training with {} GPUs...".format(G))

         # we'll store a copy of the model on *every* GPU and then combine
         # the results from the gradient updates on the CPU
         with tf.device("/cpu:0"):
             # initialize the model
             model = get_VGG_Like_model()

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
        rescale=1 / 255,
        samplewise_center=True,
        samplewise_std_normalization=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      samplewise_center=True,
                                      samplewise_std_normalization=True)

    # Training new model
    ts = str(int(time.time()))
    model_name = 'VGG16_like'
    run_name = 'model={}-batch_size={}-ts={}'.format(model_name, batch_size, ts)
    tensorboard_loc = os.path.join(TENSORBOARD_PATH, run_name)
    checkpoint_loc = os.path.join(MODELS_PATH, 'model-{}-weights.h5'.format(ts))

    earlyStopping = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                  patience=5,          # number of epochs with no improvement after which training will be stopped
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
        directory=train_path,
        target_size=(rn, cn),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        directory=valid_path,
        target_size=(rn, cn),
        batch_size=batch_size,
        class_mode='categorical')

    model_json = model.to_json()
    with open(MODELS_PATH + "model_vgg01_linux.json", "w") as json_file:
        json_file.write(model_json)
        print("save model to disk")


    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_ids) // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_ids) // batch_size,
        callbacks=callbacks_list)

    #model.save_weights('frist_try.h5')
    # save the trained model
    #model_json = model.to_json()
    #with open("model_first_try.json", "w") as json_file:
    #    json_file.write(model_json)
    #    print("save model to disk")


if __name__ == '__main__':
    train_classification()
    print('training done!!')
