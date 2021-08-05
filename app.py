import os
import numpy
import pickle
import tensorflow
from flask import Flask,render_template,request
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Function to load and transform it as required
def read_image(filename):
    img = load_img(filename,target_size=(224,224))
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    img = img.astype('float32')
    img = img / 255.0
    return img
# No Scaling 
def read_image_noscale(filename):
  img = load_img(filename,target_size=(224,224))
  img = img_to_array(img)
  img = img.reshape(1,224,224,3)
  return img
# HeartRisk Analyser Model
file=open('trained_files/heartmodel.pkl','rb')
clf =pickle.load(file)
# Home page
@app.route("/", methods=['GET','POST'])
def home():
  return render_template('home.html')
@app.route("/home.html",methods=['GET','POST'])
def home1():
  return render_template('home.html')
# Landing page 
@app.route("/landing.html",methods=['GET','POST'])
def landing():
  return render_template('landing.html')
# index --> skin lesion classification 
@app.route("/index.html", methods=['GET','POST'])
def landingpage():
  return render_template('index.html')
# index1 --> Tuberculosis classifier
@app.route("/index1.html", methods=['GET','POST'])
def landingpage1():
  return render_template('index1.html')
# index2 --> Heart disease Risk predictor
@app.route("/HeartDiseaseClassifier.html", methods=['GET','POST'])
def landingpage2():
  return render_template('HeartDiseaseClassifier.html')

@app.route("/diagnose", methods = ['GET','POST'])
def diagnose():
  if request.method == 'POST':
    file = request.files['file']
    if file and allowed_file(file.filename):
      filename = file.filename
      file_path_rt = os.path.join('static/skin_images', filename)
      file.save(file_path_rt)
      images = read_image_noscale(file_path_rt)
      # Predict the class of an image
      model = load_model('trained_files/ReslesionNet_v3gpu.h5')
      class_prediction_mb = model.predict(images)
      product_bm = class_prediction_mb *100
      class_values_mb = {'Actinic Keratosis':product_bm[0][0],'Basal Cell Carcinoma':product_bm[0][1],
      'Dermatofibroma':product_bm[0][2],'Melanoma':product_bm[0][3],'SQ.Cell Carcinoma':product_bm[0][4],
      'Soborroeic Keratosis':product_bm[0][5]}
      product_bm = class_values_mb
      #product_bm = numpy.argmax(class_prediction_mb.max())
      return render_template('result.html', product = product_bm, user_image = file_path_rt)
  
  return render_template('result.html')

@app.route('/predict', methods =['POST'])
def predict():
  features = [float(i) for i in request.form.values()]
  array_features = [numpy.array(features)]
  prediction = clf.predict(array_features)
  output = prediction
  if output == 1:
    return render_template('HeartDiseaseClassifier.html',result = 'Congratulations.. You have a Strong Heart ! You are not Likely to have Heart Diseases!')
  else:
    return render_template('HeartDiseaseClassifier.html',result = 'Oops !! You are likely to have heart disease! Consult a Doctor')

if __name__ == "__main__":
  app.run(host='0.0.0.0',port=8080)