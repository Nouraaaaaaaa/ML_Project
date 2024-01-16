import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))

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
    print(output)
    response=''
    if(output == 0.0):
        response='retained'
    else:
        response='churned'
    return render_template('index.html', response=response)



if __name__ == "__main__":
    app.run(debug=True)