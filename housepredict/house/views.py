from django.shortcuts import render
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def index(request):
    return render(request,'index.html',{})
def predict(request):
    return render(request,'predict.html')
def result(request):
    data = pd.read_csv(r"C:\house\USA_Housing.csv")
    data= data.drop("Address", axis=1)
    x = data.drop("Price", axis=1)
    y = data['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    model = LinearRegression()
    model.fit(x_train, y_train)
    val1 = float(request.GET['1'])
    val2 = float(request.GET['2'])
    val3 = float(request.GET['3'])
    val4 = float(request.GET['4'])
    val5 = float(request.GET['5'])


    pred = model.predict(np.array([val1, val2, val3, val4, val5]).reshape(1,-1))
    pred=round(pred[0])
    price="The  Predicted Price Is $"+str(pred)
    return render(request,'predict.html',{"result2":price})

# Create your views here.
