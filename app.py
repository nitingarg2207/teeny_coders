import pickle
import numpy as np
from flask import Flask,render_template,request
app = Flask(__name__)

@app.route('/')
def hello_world():
    # data=np.array([63,1,3,145,233,1,0,150,0,2.3,0,0,1])
    # data=data.reshape(1,-1)
    return render_template('index.html')
    # return render_template('index.html')
# @app.route('/predict', methods=['GET','POST'])
# def disease_prediction():
#     # data=request.form.email
#     if request.method=='POST':
#         # a = request.form
#         # n=np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
#         # n=n.reshape(1,-1)
#         # loaded_model = pickle.load(open('prediction_model.pkl','rb'))
#         data=request.form
#         age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal=data
#         n=np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
#         n=n.reshape(1,-1)
#         print(data)
#         with open('prediction_model.pkl', 'rb') as handle:
#             loaded_model = pickle.load(handle)
#         # arr=np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1)
#         x=loaded_model.predict(n)
#         x=int(x[0])
#         return render_template('result.html',resultat=x)
#     return render_template('disease_prediction.html')
#         # return "hello world"

if __name__=="__main__":
    app.run()