## 라이브러리를 불러오고, 이후에 pkl 파일을 불러오기
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

print(os.getcwd() )

model = pickle.load(open('./wine_flask_test/wine_model/wine_base.pkl', 'rb'))
print(model)

app = Flask(__name__)

## flask 앱의 루트 디렉터리 초기화
@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']

    arr = np.array([[data1, data2, data3, data4, data5, data6]])
    pred = model.predict(arr)
    return render_template('second.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)