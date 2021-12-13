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
    factors = {
      "alcohol_reference": 0,
      "drug_reference": 0,
      "use_of_alcohol": 0,
      "use_of_drugs_and_alcohol": 0,
      "simulated_gambling": 0,
      "animated_blood": 0,
      "mild_blood": 0,
      "blood": 0,
      "blood_and_gore": 0,
      "mild_cartoon_violence": 0,
      "cartoon_violence": 0,
      "mild_fantasy_violence": 0,
      "fantasy_violence": 0,
      "violence": 0,
      "mild_violence": 0,
      "intense_violence": 0,
      "mild_language": 0,
      "language": 0,
      "strong_language": 0,
      "mild_lyrics": 0,
      "lyrics": 0,
      "crude_humor": 0,
      "mature_humor": 0,
      "partial_nudity": 0,
      "nudity": 0,
      "sexual_content": 0,
      "strong_sexual_content": 0,
      "mild_suggestive_themes": 0,
      "suggestive_themes": 0,
      "sexual_themes": 0,
    }
    formdata = dict(request.args)
    for key, value in formdata.items():
      if value == 'on':
        factors[key] = 1

    X = dv.transform([factors])
    print(X)
    rating = model.predict(X)
    result = {
        'rating': rating[0],
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
