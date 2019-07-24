import os
import glob
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dense,Dropout,Flatten,Conv2D
from keras.models import Sequential
from sklearn.utils import class_weight


class Transfer_Learning_TCBB_2class:
    def __init__(self,train_dir,valid_dir,img_width,img_height,batch_size,
                 bottleneck_output_path,epochs,model_output_path):
        self.train_dir=train_dir
        self.valid_dir=valid_dir
        self.img_width=img_width
        self.img_height=img_height
        self.batch_size=batch_size
        self.bottleneck_output_path=bottleneck_output_path
        self.epochs=epochs
        self.model_output_path=model_output_path

    ## count the number of training and valid images
    def cal_train_validation_size(self):
        train_num = 0
        valid_num = 0

        train_dir=self.train_dir
        valid_dir=self.valid_dir

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

    def save_bottleneck_features_labels(self,train_num,valid_num):
        train_dir=self.train_dir
        valid_dir=self.valid_dir
        img_width=self.img_width
        img_height=self.img_height
        batch_size=self.batch_size
        bottleneck_output_path=self.bottleneck_output_path

        # we only load convolutional layers->the last layer has a shape of 7x7x512
        model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

        datagen = ImageDataGenerator(rescale=1. / 255)

        # transform training images
        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        train_labels = np.zeros(shape=(train_num,2)) # 2 subclass in our tcbb case
        i = 0
        for inputs_batch, label_batch in train_generator:
            train_labels[i * batch_size:(i + 1) * batch_size] = label_batch
            i += 1
            if i * batch_size >= train_num:
                break

        train_bottleneck_features = model.predict_generator(train_generator, steps=np.ceil(float(train_num) / batch_size))

        # trainN//batch_size return an int integer
        # trainN/btach_size return a float number
        np.savez(open(bottleneck_output_path+'train_bottleneck.npz', 'wb'), train_feats=train_bottleneck_features, train_label=train_labels)

        # transform validation images
        valid_generator = datagen.flow_from_directory(
            valid_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        valid_labels = np.zeros(shape=(valid_num,2)) # 2 subclass in our tcbb case
        i = 0
        for inputs_batch, label_batch in valid_generator:
            valid_labels[i * batch_size:(i + 1) * batch_size] = label_batch
            i += 1
            if i * batch_size >= valid_num:
                break

        valid_bottleneck_features = model.predict_generator(valid_generator, steps=np.ceil(float(valid_num) / batch_size))

        np.savez(open(bottleneck_output_path+'valid_bottleneck.npz', 'wb'), valid_feats=valid_bottleneck_features, valid_label=valid_labels)

    def train_top_layers(self):

        batch_size=self.batch_size
        epochs=self.epochs
        bottleneck_output_path=self.bottleneck_output_path
        model_output_path=self.model_output_path

        train_saved = np.load(open(bottleneck_output_path+'train_bottleneck.npz', 'rb'))
        train_data = train_saved['train_feats']
        train_labels = train_saved['train_label']

        valid_saved = np.load(open(bottleneck_output_path+'valid_bottleneck.npz', 'rb'))
        valid_data = valid_saved['valid_feats']
        valid_labels = valid_saved['valid_label']

        model = self.build_top_layers(train_data.shape[1:])


        print(model.summary())

        ## try add class weight
        #y_integers=np.argmax(train_labels,axis=1)
        #class_weights=class_weight.compute_class_weight('balanced',np.unique(y_integers),y_integers)
        #d_class_weights=dict(enumerate(class_weights))

        #sample_weights=class_weight.compute_sample_weight('balanced',train_labels)

        ## try to add sample weight

        sgd = optimizers.SGD(lr=0.0003, clipvalue=0.5)
        #rmsprop=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        #adam=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(valid_data, valid_labels))

        model.save_weights(model_output_path+'TL_model_weights_v00.h5')

        # save the trained model
        model_json = model.to_json()
        with open(model_output_path+'TL_model_v00.json', 'w') as json_file:
            json_file.write(model_json)
            print('saved model to disk!!!!')

    def build_top_layers(self,bottleneck_shape):
        model = Sequential()
        model.add(Conv2D(128, (1, 1), input_shape=bottleneck_shape, activation='relu'))
        #model.add(Conv2D(128, (1, 1), activation='relu'))
        #model.add(Flatten(input_shape=bottleneck_shape))  # input shape=(7,7,512)
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(Dense(512,activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        return model
