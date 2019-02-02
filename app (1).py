import os
from flask import Flask, request, jsonify


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import requests
import json
import pandas as pd
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None


# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global model
    global graph
    model = pickle.load(open('predictingflightdelays.sav','rb'))


load_model()

def prepare_csv(filename):
    df=pd.read_csv(filename)
    return df.head(1)
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the mnist
            # format of 28x28 pixels
            

            # Convert the 2D image to an array of pixel values
            image_array= prepare_csv(filepath)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
           # global graph
            #with graph.as_default():

                # Use the model to make a prediction
            predicted_digit = model.predict(image_array)[0]
            data["prediction"] = str(predicted_digit)

            # indicate that the request was a success
            data["success"] = True

            return jsonify(data)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
