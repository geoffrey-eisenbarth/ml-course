from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


with open('model.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)
with open('options.bin', 'rb') as f:
    options = pickle.load(f)


app = Flask('laptop_price')


@app.route('/')
def index():
    response = render_template(
        'index.html',
        title='Laptop Prices',
        **options,
    )
    return response


@app.route('/predict/')
def predict():
    laptop = dict(request.args)
    ln_weight = np.log(float(laptop.pop('weight')))
    laptop['ln_weight'] = ln_weight

    X = dv.transform([laptop])
    ln_price = model.predict(X)

    result = {
        'price': round(np.exp(ln_price)[0], 2),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
