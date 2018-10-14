import cv2
import numpy as np
import pandas as pd
from skimage import io
import skimage.transform
from keras.models import load_model
import os
import pickle

def upsample_skimage1(factor, input_img):

    # Pad with 0 values, similar to how Tensorflow does it.
    # Order=1 is bilinear upsampling
    return skimage.transform.rescale(input_img,
                                     factor,
                                     mode='constant',
                                     cval=0,
                                     order=1)

def label_subimages(img_path, box_len=112, factor=2):
    img = cv2.imread(img_path)
    pic_name = os.path.splitext(os.path.basename(img_path))[0]
    num_x, num_y = img.shape[1]//box_len, img.shape[0]//box_len # y comes first in tuple
    output = np.zeros((num_x, num_y))

    pad_x, pad_y = img.shape[1]%box_len//2, img.shape[0]%box_len//2 #get start positions padding

    for i in range(num_x):
        for j in range(num_y):
            sub_img = img[pad_y+j*box_len:pad_y+(j+1)*box_len,
                                  pad_x+i*box_len:pad_x+(i+1)*box_len]
            ups = upsample_skimage1(factor, sub_img)
            #io.imsave((pic_name + str(i) + '-' + str(j) + '.png'), ups)
            im_exp = np.expand_dims(ups, axis=0)
            probs = model.predict(im_exp, batch_size=1, verbose=1) # pass into model
            output[i,j] = np.argmax(probs)
    return output

model = load_model('weeds14.hdf5')

out = label_subimages('pasture1.jpg', box_len=224, factor=1)
print(pd.Series(out.reshape(-1)).value_counts())

with open('pasture1_map224-model14.pkl', 'wb') as picklefile:
    pickle.dump(out, picklefile)
