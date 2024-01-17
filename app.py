import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_final1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = request.form.getlist('features')

        int_features = [float(feature) for feature in int_features]

        print(int_features)
        print('before prediction')
        prediction = model.predict(np.array(int_features).reshape(1, -1))

        print(prediction)
        print('after prediction')

        output = round(prediction[0], 1)
        print(output)

        response = 'retained' if prediction[0] == 0 else 'churned'

        print("Prediction Result:", response)

        return render_template('index.html', response=response)
    except Exception as e:
        return render_template('index.html', response=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)