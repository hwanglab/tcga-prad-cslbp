import os
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import VGG16
from skimage.io import imshow
import numpy as np
from scipy.io import loadmat,savemat
import glob

fold=3

if fold==1:
    Test_g6=['../fold1/valid/g6/']  # fold 1
    Test_g7=['../fold1/valid/g7/']
    Test_g8_9=['../fold1/valid/g8_9/']

    model_location='./fold02_dl/models/'+'model_vgg01_linux.json'
    model_weights='./fold02_dl/models/'+'model_weights.h5'
elif fold==2:
    Test_g6 = ['../fold2/valid/g6/']  # fold 1
    Test_g7 = ['../fold2/valid/g7/']
    Test_g8_9 = ['../fold2/valid/g8_9/']

    model_location = './fold02_dl/models/' + 'model_vgg01_linux.json'
    model_weights = './fold02_dl/models/' + 'model_weights.h5'
elif fold==3:
    Test_g6 = ['../fold3/valid/g6/']  # fold 1
    Test_g7 = ['../fold3/valid/g7/']
    Test_g8_9 = ['../fold3/valid/g8_9/']

    model_location = './fold02_dl/models/' + 'model_vgg01_linux.json'
    model_weights = './fold02_dl/models/' + 'model_weights.h5'
else:
    print('impossible!!!!!')


patientDI=loadmat('../'+'patInfo.mat')


rn=cn=256

def load_model_prediction():

    json_file = open(model_location, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights)

    pID = patientDI['patientInfo']
    tnum = 0
    cnum = 0
    pIDL = []
    pind_mean = []
    pind_maj = []
    preds = []
    for k in range(0, len(pID[:, 0])):
        pID_temp = pID[k, 0]  # patient ID
        cc_temp = pID[k, 1]  # class type
        print(pID_temp, '--', k)
        if cc_temp == 6:
            tnum += 1
            pIDL.extend(pID_temp)
            pred = tmb_prediction(model, pID_temp[0], Test_g6)

            preds.extend([pred])


        elif cc_temp == 7:
            tnum += 1
            pIDL.extend(pID_temp)
            pred = tmb_prediction(model, pID_temp[0], Test_g7)

            preds.extend([pred])


        elif cc_temp ==8:
            tnum += 1
            pIDL.extend(pID_temp)
            pred = tmb_prediction(model, pID_temp[0], Test_g8_9)

            preds.extend([pred])


        else:
            print('impossible!!!!')

    # print('accuray is {}'.format(cnum/tnum))

    savemat("vgg_dl_result.mat", dict([('pID', pIDL), ('preds', preds)]))


def tmb_prediction(model,pID_temp,image_path):
    images=[]
    for kk in range(0,len(image_path)):
        images = glob.glob(image_path[kk] + pID_temp + '*.png')
        if len(images)>0:
            break

    if len(images)>0:
        print(pID_temp+'---testing!!!')

        pred=[]
        for image_name in images:
            if pID_temp in image_name:
                img=image.load_img(image_name,target_size=(rn,cn))
                x=image.img_to_array(img)

                # preprocessing
                x *= 1 / 255.0
                x -= np.mean(x, keepdims=True)
                x /= (np.std(x, keepdims=True) + 1e-6)

                x=np.expand_dims(x,axis=0)

                pred_lab=model.predict(x) # ??? the probability belong to the first class: low TMB
                pred.append(np.ndarray.tolist(pred_lab[0]))

        pred=np.asarray(pred)

        # average predictions
        pred=np.mean(pred,axis=0)

    else:
        pred=-1*np.ones((1,3))

    return pred

if __name__=='__main__':
    load_model_prediction()