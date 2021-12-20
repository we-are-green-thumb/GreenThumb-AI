# import os
# from PIL import Image
# import torch.nn as nn
# import torch.optim as optim
from torchvision import datasets, models, transforms
# import torch

# model_filename = '/content/drive/MyDrive/AI/model/checkpoint.h5'

# model = torch.load(model_filename)
# model.eval()


from flask import Flask, jsonify, request
app = Flask(__name__)

# import tensorflow as tf 
import torch
# import h5py
# transforms_test = tf.Compose([
#     tf.Resize((224, 224)),
#     tf.ToTensor(),
#     tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# model = torch.load('./checkpoint.h5')
# print(model)


import h5py
model_filename = './checkpoint.h5'
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.device('cpu')
# model = torch.load(model_filename)
# model.eval()
# print(model.eval())

h5 = h5py.File(model_filename,'r')
futures_data = h5['futures_data']  # VSTOXX futures data
options_data = h5['options_data']  # VSTOXX call option data
h5.close()


# print(torch.cuda.is_available())

# class_names = ['로즈마리', '로즈마리 마름', '유칼립투스',
#                '유칼립투스 잎마름', '장미', '장미 검은무늬병', '장미 흰가루병']

# @app.route('/predict', methods=['POST'])
# def predict():

#     # 전송된 이미지 받기
#     requestImg = request.get_json()
#     url = requestImg['imageUrl']
#     print(url)

#     # # 저장한다
#     # os.system("curl " + url + " > test.jpg")
#     # image = Image.open("test.jpg")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # image = transforms_test(image).unsqueeze(0).to(device)

#     # with torch.no_grad():
#     #     outputs = model(image)
#     #     _, preds = torch.max(outputs, 1)

#     # return class_names[preds[0]]
#     return '장미검은무늬병'


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port="5000")