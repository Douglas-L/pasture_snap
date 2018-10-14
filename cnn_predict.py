#working copy

import flask
from flask import request, render_template, send_from_directory
import numpy as np
import pandas as pd
from copy import deepcopy
from werkzeug import secure_filename
from PIL import Image

#Process image
import cv2
from skimage import io
import skimage.transform
from keras.models import load_model
import os
import pickle

#---------- MODEL IN MEMORY ----------------#

model = load_model('weeds21.hdf5') #load best model
model._make_predict_function() # fixes tensor not an element of this graph error - why?


#---------- FUNCTIONS IN MEMORY ----------------#
def upsample_skimage1(factor, input_img):

    # Pad with 0 values, similar to how Tensorflow does it.
    # Order=1 is bilinear upsampling, 3 for bicubic 
    # chose 1 for speed (~2x faster) since 3 didn't noticeably improve quality
    return skimage.transform.rescale(input_img,
                                     factor,
                                     mode='constant',
                                     cval=0,
                                     order=1)

def label_subimages(img_path, box_len=112, factor=2):
    img = cv2.imread(img_path) # read img as array
    #pic_name = os.path.splitext(os.path.basename(img_path))[0]
    num_x, num_y = img.shape[1]//box_len, img.shape[0]//box_len # y comes first in tuple
    output = np.zeros((num_y, num_x)) #create matrix to fit over original image
    output2 = np.zeros((num_y, num_x))
    
    pad_x, pad_y = img.shape[1]%box_len//2-1, img.shape[0]%box_len//2-1 #get start positions padding
    #pad both sides, -1 to be safe because of indexing
    
    list_probs = []
    for i in range(num_y): #loop over rows
        for j in range(num_x): #loop over columns of each row
            sub_img = img[pad_y+i*box_len:pad_y+(i+1)*box_len,
                                  pad_x+j*box_len:pad_x+(j+1)*box_len]
            ups = upsample_skimage1(factor, sub_img)
            #print(i, j, ups.shape)
            #io.imsave((pic_name + str(i) + '-' + str(j) + '.png'), ups)
            im_exp = np.expand_dims(ups, axis=0)
            probs = model.predict(im_exp, batch_size=1, verbose=0) # pass into model, 
            # 0 to reduce spam until increase batch_size - maybe flatten for efficiency
            list_probs.append(probs)
            output[i,j] = np.argmax(probs) # fill in matrix with prediction
            output2[i,j] = np.amax(probs) # make another matrix to be able to show confidence level?
    preds = output.reshape(-1)
    total = len(preds)
    pct_forbs = len(preds[preds == 0]) / total
    pct_grass = len(preds[preds == 1]) / total
    pct_ground = len(preds[preds == 2]) / total
    pct_weeds = len(preds[preds == 3]) / total   
#     df = pd.DataFrame(output)
#     df2 = pd.DataFrame(output2)
#     out = df.to_html(header=False, index=False, escape=False)
#     out2 = df2.to_html(header=False, index=False, escape=False)
    accum_prob = np.sum(np.array(list_probs), axis=0)
    d = {0:'Forb', 1:'Grass', 2:'Ground', 3:'Weed'}
    pred_cls = [d[k] for k in preds if k in d]
    max_probs = output2.reshape(-1).round(2).tolist()
    list_probs = np.array(list_probs).tolist()
            
    return dict( sacross=num_x, sdown=num_y, pad_x=pad_x, pad_y=pad_y, sub_images=total, grass=pct_grass, ground=pct_ground, forbs=pct_forbs, weeds=pct_weeds,pred_mat=pred_cls, prob_mat=max_probs, all_probs=list_probs, all_prob_mat=accum_prob)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

#
UPLOAD_FOLDER = 'static/img' #default is just 'static' - need to mkdir first
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#not sure if optimal 

@app.route("/")
def viz_page():
    return render_template('upload.html') #html files needs to be in a 'templates' folder 

@app.route("/prediction", methods=["GET","POST"])
def deep_image():
    if request.method == 'POST':
        f = request.files['image_uploads'] #get form input type 'file' with ['name'] 
        filename = secure_filename(f.filename) #turn /'s into _'s to make names safe
        sfname = os.path.join(app.config['UPLOAD_FOLDER'], filename) #make target path
        f.save(sfname) #save 
        
        #call function on saved image
        abs_sf = os.path.abspath(sfname) 
        out = label_subimages(abs_sf) 
        
        return render_template('d3test.html', pred = out, img_name = filename)


@app.route("/prediction/<filename>") #called in prediction.html to render image
def send_image(filename):
    
    return send_from_directory("static/img", filename)

#TO RUN?:
#export FLASK_APP=cnn_predict.py
#flask run --host=0.0.0.0
#enable tcp 5000 port access on AWS

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80

# if __name__ == '__main__':
#     app.run(debug = False)
# (The default website port)
# app.run(host='0.0.0.0', port=80) got on on port 5000
# app.run(debug=False)
