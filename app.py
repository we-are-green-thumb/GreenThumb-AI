import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from flask import Flask, jsonify, request
app = Flask(__name__)


transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = torch.load('checkpoint.h5')
print(model)

class_names = ['로즈마리', '로즈마리 마름', '유칼립투스',
               '유칼립투스 잎마름', '장미', '장미 검은무늬병', '장미 흰가루병']


@app.route('/predict', methods=['POST'])
def predict():

    # 전송된 이미지 받기
    requestImg = request.get_json()
    url = requestImg['imageUrl']
    print(url)

    # 저장한다
    os.system("curl " + url + " > test.jpg")
    image = Image.open("test.jpg")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]
    # return '장미검은무늬병'
