from flask import Flask, jsonify
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # return 'Hello World!'
    return '장미검은무늬병'