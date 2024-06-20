from flask import Flask, render_template , request ,url_for
import os
from PIL import Image    
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn import tree

# Create a directory in a known location to save files to.
# uploads_dir = os.path.join(app.instance_path, 'uploads')
# os.makedirs(uploads_dir, exists_ok=True)

app = Flask(__name__)
UPLOAD_FOLDER = './upImg'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/home',methods=['GET','POST'])
def home():
    if request.form.get('files') != 'POST' :
        return render_template('index.html')
   


# # A route with parameter
@app.route('/process',methods=['POST','GET'])
def process():
    if request.method == "POST":
        # if request.files:
        #     image = request.files["image"]
        #     print(image)
        #     return render_template('result.html',result = image)
        f = request.files['image']
        if f.filename == '':
            return 'not up'
        if f:  
            # f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
            from test import test2
            # a = test2(f)

            # f.save(f.filename)  
            
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
            new_path = os.path.abspath(UPLOAD_FOLDER)
            print(new_path)
            a= test2(new_path) #get image and taste variable
            print(a)
            filename = new_path + f.filename
            return render_template('test.html', taste=a ,gg=filename,pp=new_path,name=f.filename)
        return 'not'

    # if request.method == 'POST' :
    #     a = request.form.get('image')
    #     # list_of_images = os.listdir(path_of_images)
    # return render_template('page_404.html')



if __name__ == '__main__':
    app.run(debug=True)

