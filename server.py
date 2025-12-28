from flask import Flask, request, jsonify
import numpy as np
from src.model import NeuralNetwork

app = Flask(__name__)
nn = NeuralNetwork(2, 2, 1)

try:
    nn.load("brain.npz")
    print("Successfully loaded 'brain.npz'")
except:
    print("Could not find 'brain.npz'. Please run src/model.py first!")

@app.route('/')
def home():
    return "Neural Network Server is Running!"

@app.route('/predict', methods=['GET'])
def predict():
    a = float(request.args.get('a', 0))
    b = float(request.args.get('b', 0))

    input_data = np.array([[a, b]])

    prediction = nn.forward(input_data)
    result = float(prediction[0][0])

    return jsonify({
        'input': [a, b],
        'prediction': result,
        'interpretation': "True" if result > 0.5 else "False"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)