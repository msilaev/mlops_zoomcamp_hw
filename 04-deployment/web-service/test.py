import predict
import requests


ride = {
    "PULocationID": 1,
    "DOLocationID": 2,
    "trip_distance": 3.5,
}

response = requests.post('http://localhost:9696/predict', json=ride)
print(response.json())
#features = predict.prepare_features(ride)

#pred = predict.predict(features)
#print(pred)