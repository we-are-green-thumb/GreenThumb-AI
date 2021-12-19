# from flask import Flask #간단히 플라스크 서버를 만든다

# import urllib.request

# app = Flask(__name__)

# @app.route("/tospring")
# def spring():
    
#     return "test"
    
    
# if __name__ == '__main__':
#     app.run(debug=False,host="127.0.0.1",port=5000)


# from flask import Flask, request, render_template
# from dao import EmpDAO

# app = Flask(__name__)


# @app.route('/', methods= ['get'])
# def get():
#     return render_template('reqres.html')

# #사번으로 사원 검색
# @app.route('/getempno', methods = ['post'])
# def getoneemp(empno):
#     return EmpDAO().select_empno(request.form.get('empno'))


# if __name__ == '__main__':
#     app.run(debug=True)


#---------------------------------------------------------------------------------------
# import io
# from flask_ngrok import run_with_ngrok
# from flask import Flask, jsonify, request


# # 이미지를 읽어 결과를 반환하는 함수
# def get_prediction(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes))
#     image = transforms_test(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(image)
#         _, preds = torch.max(outputs, 1)
#         imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

#     return class_names[preds[0]]


# app = Flask(__name__)


# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # 이미지 바이트 데이터 받아오기
#         file = request.files['file']
#         image_bytes = file.read()

#         # 분류 결과 확인 및 클라이언트에게 결과 반환
#         class_name = get_prediction(image_bytes=image_bytes)
#         print("결과:", {'class_name': class_name})
#         return jsonify({'class_name': class_name})