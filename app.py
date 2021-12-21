import io
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch

import torch.nn as nn
import numpy as np
import urllib.request
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import Image

from flask import Flask, jsonify, request


model = torch.load('./model/checkpoint.pth', map_location=torch.device("cpu"))
device = torch.device("cpu")
model.eval()



class_names=['로즈마리', '로즈마리 마름', '유칼립투스', '유칼립투스 잎마름', '장미', '장미 검은무늬병', '장미 흰가루병']

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    # image = Image.open(io.BytesIO(image_bytes))
    image = Image.open('./rose.jpg')
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        plt.imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])
        print(class_names[preds[0]])

    return class_names[preds[0]]

def url_to_image(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype='uint8')
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  return image



app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 바이트 데이터 받아오기
    # requestImg = request.get_json()
    # urls = requestImg['imageUrl']
    # url = urls[0]
    # print(url)
    
    # response = url_to_image(url)
    
    # 분류 결과 확인 및 클라이언트에게 결과 반환
    class_name = get_prediction('./rose.jpg')
    print(class_name)
    print("결과:", {'class_name': class_name})
    return jsonify({'class_name': class_name})
    # return '장미검은무늬병';

if __name__ == "__main__":
    app.run(debug=True)
