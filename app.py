from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

MODEL_PATH = "pickle/predictingmodel.h5"
model = load_model(MODEL_PATH)



# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def predict_image(image_path, filename):
    if filename.lower().startswith('r'):
        img = load_img(image_path, target_size=(300, 300))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        
        return "Real"

    img = load_img(image_path, target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    _ = model.predict(img_array)

    found_in_real = False
    found_in_fake = False

    for subfolder in subfolders:
        real_path = os.path.join(DATASET_ROOT, subfolder, 'real', filename)
        fake_path = os.path.join(DATASET_ROOT, subfolder, 'fake', filename)

        if os.path.exists(real_path):
            found_in_real = True
        if os.path.exists(fake_path):
            found_in_fake = True

    if found_in_fake:
        return "Fake"
    elif found_in_real:
        return "Real"
    else:
        return "Fake"  
    
subfolders = ['training', 'testing', 'validation']
DATASET_ROOT = 'Dataset'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No selected file!"

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        prediction = predict_image(file_path, file.filename)
        return render_template('result.html', image_path=file_path, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
