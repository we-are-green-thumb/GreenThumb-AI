
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from flask import Flask,  request, jsonify
import cv2
import numpy as np
import urllib.request
import numpy as np
import os
import time
from util import util


app = Flask(__name__)

 # device 객체
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#전이모델 받은 것 로딩
model = models.resnet34()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.load_state_dict(torch.load('./model/modelhospital_weights2.pth', map_location=device))
model.to(device)
model.eval()
 
class_names= ['검은무늬병', '과습', '물부족', '정상', '흰가루병']
# class_names=['장미검은무늬병', '장미 점박이응애', '장미 흰가루병']

#이미지 데이터 학습할 때와 동일하게 전처리
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/predict', methods=['POST'])
def predict():
    
    # 전송된 이미지 받기
    requestImg = request.get_json()
    url = requestImg['imageUrl']
    
    os.system("curl " + url[0] + " > test.jpg")
    start = time.time()
    # 이미지 다운로드 시간 체크
    print(time.time() - start)

    # 저장 된 이미지 확인
    img = Image.open("test.jpg") 
    # testimg = url[0].save("./test.jpg")
    # print(url[0])
    # urls = url[0]
    
    image = transforms_test(img).unsqueeze(0).to(device)
    print(image)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        # imshow(image.cpu().data[0], '예측 결과: ' + class_names[preds[0]])
        print(class_names[preds[0]])
        disease = class_names[preds[0]]
        print(type(disease))
    
    # result = ''
    # for char in disease :
    #     print(char)
    #     result += char;
    
    # return  util.join_jamos(result)
    return disease



if __name__ == "__main__":
    app.run(debug=True)
