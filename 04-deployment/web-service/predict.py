import pickle

from flask import Flask, request, jsonify

with open("lin_reg.bin", "rb") as f:
    (dv, model) = pickle.load(f)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']

    return features

def predict(features):
    X = dv.transform(features)
    y_pred = model.predict(X)
    return y_pred[0]


app = Flask('duration-predictor')

@app.route('/predict', methods=['POST'])
def precit_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    result = {
        'duration': pred
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run( debug=True,  host='0.0.0.0', port=9696)