from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
app=Flask(__name__)
with open('LoanPrediction.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/result',methods=['post'])
def result():
    data=[]
    gender = int(request.form.get('gender'))
    married = int(request.form.get('married'))
    dependence = int(request.form.get('dependence'))
    education = int(request.form.get('education'))
    self_employed = int(request.form.get('self_employed'))
    montly_income = float(request.form.get('montly_income'))
    co_applicant_income = float(request.form.get('Co_applicant_income'))
    loan_amount= float(request.form.get('loan_amount'))
    credit_history= float(request.form.get('credit_history'))
    property_area= int(request.form.get('property_area'))

    data.extend([gender,married,dependence,education,self_employed,montly_income,co_applicant_income,loan_amount,credit_history,property_area])
    print(data)
    data=np.array(data)
    data = data.reshape(1, -1)
    output=model.predict(data)
    data=output[0]


    return render_template('result.html',data=data)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
