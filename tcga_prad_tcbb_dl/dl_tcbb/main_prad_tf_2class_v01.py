
import os
import time
import scipy.misc
from keras.models import model_from_json
from keras.applications import VGG16

from Transfer_Learning_TCBB_2class import Transfer_Learning_TCBB_2class

save_bottleneck_predictions=1
training=1

model_path = './two_class/fold03_tl/'
bottleneck_output_path = './two_class/fold03_tl/'

if __name__=='__main__':

    if training==1:
        # parameter settings
        train_dir = '../two_class/fold3/train/'
        valid_dir = '../two_class/fold3/valid/'
        batch_size = 64
        img_width, img_height = 224, 224
        epochs = 100


        tl_model = Transfer_Learning_TCBB_2class(train_dir, valid_dir, img_width, img_height, batch_size, bottleneck_output_path,
                                     epochs, model_path)

        if save_bottleneck_predictions == 1:
            train_num, valid_num=tl_model.cal_train_validation_size()
            tl_model.save_bottleneck_features_labels(train_num,valid_num)

        tl_model.train_top_layers()


        #cd projects/tcga_prad/tcga_prad_tcbb_dl/dl_tcbb
