import os
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import VGG16
#from skimage.io import imshow
import numpy as np
from scipy.io import loadmat,savemat
import glob

Test_g7=['../two_class/fold3/valid/g7/']  # fold 1
Test_g8_9=['../two_class/fold3/valid/g8_9/']

model_location='./two_class/fold03_tl/'+'TL_model_v00.json'
model_weights='./two_class/fold03_tl/'+'TL_model_weights_v00.h5'

patientDI=loadmat('../'+'patInfo.mat')


rn=cn=224

def load_model_prediction():
    model_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
        if cc_temp == 6 or cc_temp==7:

            tnum += 1
            pIDL.extend(pID_temp)
            pred = tmb_prediction(model_base,model, pID_temp[0], Test_g7)

            preds.extend([pred])


        elif cc_temp ==8:
            tnum += 1
            pIDL.extend(pID_temp)
            pred = tmb_prediction(model_base,model, pID_temp[0], Test_g8_9)

            preds.extend([pred])


        else:
            print('impossible!!!!')

    # print('accuray is {}'.format(cnum/tnum))

    savemat("vggresult.mat", dict([('pID', pIDL), ('preds', preds)]))


def tmb_prediction(model_base,model,pID_temp,image_path):
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
                #x -= np.mean(x, keepdims=True)
                #x /= (np.std(x, keepdims=True) + 1e-6)

                x=np.expand_dims(x,axis=0)
                x2=model_base.predict(x)

                pred_lab=model.predict(x2) # ??? the probability belong to the first class: low TMB
                pred.append(np.ndarray.tolist(pred_lab[0]))

        pred=np.asarray(pred)

        # average predictions
        pred=np.mean(pred,axis=0)

    else:
        pred=-1*np.ones((1,2))

    return pred

if __name__=='__main__':
    load_model_prediction()