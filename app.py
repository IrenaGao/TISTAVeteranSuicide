import numpy as np
from flask import Flask, request, jsonify, render_template
import flask
import pickle
# import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from google.colab import drive
# drive.mount('/content/drive')
# path = '/content/drive/MyDrive/synthetic.csv'

app = Flask(__name__)
logReg = pickle.load(open('logReg.pkl', 'rb'))
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/friendsandfam')
def friendsandfam():
    return render_template('friendsandfam.html')

@app.route('/logPredict',methods=['GET', 'POST'])
def logPredict():

    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    int_features.append(int_features[0]*int_features[-2])
    final_features = np.array(int_features)
    prediction = round(logReg.predict_proba(final_features.reshape(1,-1))[0][1], 2)
    prediction *= 100

    return render_template('friendsandfam.html', prediction_text_log='Based on your responses, this model predicts a suicide risk of {}%.'.format(prediction))

@app.route('/lstmPredict',methods=['GET','POST'])
def lstmPredict():
    MAX_SEQUENCE_LENGTH = 280
    input_data = [x for x in request.form.values()]
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(input_data)
    # Pad sequences to the same length.
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    pred = loaded_model.predict(data)
    pred = np.round(pred.flatten())[0]
    print(pred)
    res = "high risk" if pred <= 0.3 else "low risk"
    return render_template('friendsandfam.html', prediction_text_twitter='Based on your tweet, this model predicts a {} of suicide.'.format(res))
    # return render_template('index.html', prediction_text='Your Result is {}'.format("hi"))

@app.route('/getData',methods=['GET'])
def getData():
    return

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)