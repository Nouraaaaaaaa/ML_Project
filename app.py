import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_last.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print("---------",prediction)
    output = round(prediction[0], 1)
    print(type(output))
    response=''
    if(output == 0.0):
        response='churned'
    else:
        response='retained'
    return render_template('index.html', prediction_text='the client will be '+response)



if __name__ == "__main__":
    app.run(debug=True)