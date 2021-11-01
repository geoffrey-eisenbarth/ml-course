from flask import Flask, request, jsonify
import numpy as np
import pickle


with open('model.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)


app = Flask('laptop_price')


@app.route('/predict', methods=['POST'])
def predict():
    laptop = request.get_json()

    X = dv.transform([laptop])
    ln_price = model.predict(X)

    result = {
        'price': np.exp(ln_price),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
