
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from flask import Flask,  request
import cv2
import numpy as np
import urllib.request
import numpy as np
import os
import time

app = Flask(__name__)

 # device 객체
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#전이모델 받은 것 로딩
model = models.resnet34()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
# model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model.load_state_dict(torch.load('./model/modelhospital_weights.pth', map_location='cuda:0'))

model.load_state_dict(torch.load('./model/modelhospital_weights.pth', map_location=device))
model.to(device)
model.eval()

class_names=['장미검은무늬병', '장미 점박이응애', '장미 흰가루병']

#이미지 데이터 학습할 때와 동일하게 전처리
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# #이미지 확인 용도로 사용함
# def imshow(input):
#     # torch.Tensor -> numpy
#     input = input.numpy().transpose((1, 2, 0))
#     # 이미지 정규화 해제하기
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     input = std * input + mean
#     input = np.clip(input, 0, 1)
#     # 이미지 출력
#     plt.imshow(input)
#     # plt.title(title)
#     plt.show()





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

    image = Image.open('test.jpg')
    # image = Image.open(urls)
    
    image = transforms_test(image).unsqueeze(0).to(device)
    print(image)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        # imshow(image.cpu().data[0], '예측 결과: ' + class_names[preds[0]])
        print(class_names[preds[0]])
        disease = class_names[preds[0]]

    return disease



if __name__ == "__main__":
    app.run(debug=True)
