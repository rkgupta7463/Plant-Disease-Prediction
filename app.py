from flask import Flask, redirect, url_for, render_template, request,send_from_directory,jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the "uploads" directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


### initilizing the model
plant_model=tf.keras.models.load_model('model/Plant_disase_model.h5')
classify_plant_human=tf.keras.models.load_model('model/plant_human_model.h5')

## Prediction function
def prediction_class(img,model):
    # Open the image from the file stream using PIL
  img_pil = Image.open(img)
  # Convert the image to RGB mode (in case it's in RGBA)
  img_pil = img_pil.convert("RGB")  
  # Resize the image to the expected input size of the model (e.g., 255x255)
  img_test_resized = img_pil.resize((255, 255))
  targeted_img_3=tf.keras.preprocessing.image.img_to_array(img_test_resized)
  targeted_img_3 = np.expand_dims(targeted_img_3, axis=0)
  reslut_3=model.predict(targeted_img_3)

  class_name={0:'Healthy', 1:'Powdery', 2:'Rust'}

  arr = np.array(reslut_3)

  # Find the index of the maximum value
  max_index = np.argmax(arr)

  class_pred=class_name[max_index]

  disease_description={"Healthy":"A thriving plant displays robust growth, vibrant leaves, sturdy stems, and resists pests and diseases. It thrives under optimal conditions: sufficient water, well-drained soil, sunlight, suitable temperature, balanced soil pH, and essential nutrients.","Powdery":"Powdery mildew is a common fungal disease that affects a wide variety of plants, causing white, powdery spots on leaves, stems, and flowers. It is caused by a group of fungi that belong to the order Erysiphales. Powdery mildew can be a problem for both ornamental and vegetable plants, and it can cause significant damage to crops.","Rust":"Rust is a common fungal disease that affects a wide variety of plants, causing orange, yellow, brown, or red pustules on leaves, stems, and fruits. It can reduce plant vigor and yield, and in some cases, can kill the plant."}
  disease_article_link={"Healthy":"https://whyfarmit.com/spruce-tree-indoors/","Powdery":"www.almanac.com/pest/powdery-mildew","Rust":"https://www.planetnatural.com/pest-problem-solver/plant-disease/common-rust/"}

  pred_des=disease_description[class_pred]
#   print(pred_des)
  pred_link=disease_article_link[class_pred]
#   print(pred_link)
  context={"Class name":class_pred,"pred_des":pred_des,"pred_link":pred_link}

  return context

def plant_human_classification(img,model):
  img_pil = Image.open(img)
  # Convert the image to RGB mode (in case it's in RGBA)
  img_pil = img_pil.convert("RGB")  
  # Resize the image to the expected input size of the model (e.g., 255x255)
  img_test_resized = img_pil.resize((255, 255))
  targeted_img_3=tf.keras.preprocessing.image.img_to_array(img_test_resized)
  targeted_img_3 = np.expand_dims(targeted_img_3, axis=0)
  reslut_3=model.predict(targeted_img_3)

  class_name={0 : 'human', 1 : 'plants_leaf'}
  if reslut_3 >= 0.55:
     res=np.ceil(reslut_3)
    #  print("ceil func:- ",np.ceil(reslut_3))
  else:
     res=np.floor(reslut_3)
    #  print("floor func:- ",np.floor(reslut_3))
  arr = np.array(res)
  idx=arr[0,0]
  class_pred=class_name[np.abs(idx)]
  
  context={"predicted Value" : reslut_3, "Predicted class name:": class_pred}

  return idx


@app.route('/', methods=['GET', 'POST'])
def home():
    result=None
    if request.method == 'POST':
        location = request.form.get('location')
        img = request.files['img']

        if img:
            ## image saving 
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))

            ## prediction function
            p_res=plant_human_classification(img,classify_plant_human)
            # print(p_res)

            if p_res==1:
              result=prediction_class(img,plant_model)
              # print(result)
              return render_template('index.html', location=location, img=img.filename,result=result)
            else:
                message={"message":"This image is not acceptable, you can only upload the plant leafs🌿🍃🍀🍂🥬!"}
                # print(jsonify(message))
                return jsonify(message)
    return render_template('index.html')



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(debug=True, host="0.0.0.0")
