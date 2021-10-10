from flask import Flask, request, jsonify
import pickle


with open('model1.bin', 'rb') as f:
  model = pickle.load(f)
with open('dv.bin', 'rb') as f:
  dv = pickle.load(f)


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
  customer = request.get_json()

  X = dv.transform([customer])
  not_churn, will_churn = model.predict_proba(X)[0]

  result = {
    'churn_probability': float(will_churn),
    'churn': bool(will_churn >= 0.5),
  }
  return jsonify(result)


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=9696)
