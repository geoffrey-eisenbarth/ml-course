from flask import Flask, request, jsonify, render_template
import pickle


with open('model.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)


app = Flask('esrb_prediction')


@app.route('/')
def index():
    response = render_template(
        'index.html',
        title='ESRB Prediction',
    )
    return response


@app.route('/predict/')
def predict():
    factors = dict(request.args)
    X = dv.transform([factors])
    rating = model.predict(X)

    result = {
        'rating': rating,
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
