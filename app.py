import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
logReg = pickle.load(open('logReg.pkl', 'rb'))
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/logPredict',methods=['GET', 'POST'])
def logPredict():

    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = logReg.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Suicide Likelihood is {}'.format(output))

@app.route('/lstmPredict',methods=['GET','POST'])
def lstmPredict():
    MAX_SEQUENCE_LENGTH = 280
    input_data = [x for x in request.form.values()]
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(input_data)
    #Pad sequences to the same length.
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    pred = loaded_model.predict(data)
    pred = np.round(pred.flatten())[0]
    res = "High Risk, help is available" if pred <= 0.3 else "Low Risk"
    return render_template('index.html', prediction_text='Your Result is {}'.format(res))

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