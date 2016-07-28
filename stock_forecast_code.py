# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 22:26:05 2016

@author: Henry
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:24:17 2016

@author: Henry
"""

   
import numpy
data=numpy.loadtxt(fname='C:\\Users\\Henry\\Desktop\\machine_learning_challenge\\stock_training_data.csv',delimiter=',')

test_data=numpy.loadtxt(fname='C:\\Users\\Henry\\Desktop\\machine_learning_challenge\\test_data.csv',delimiter=',')

myData=numpy.zeros((40,9))
data_temp=numpy.zeros((1,9))

output=numpy.zeros((50,1))

TOP=11
LOW=10

myTag=[0 for i in range(40)]

for i in range(40):
   myTag[i]=data[TOP-1,0]
   data_temp=data[LOW:TOP,1:10]
   
   myData[i, ]=numpy.reshape(data_temp,9)
   TOP=TOP+1
   LOW=LOW+1
   
from sklearn import svm  
clf=svm.SVR()
clf.fit(myData, myTag) 
pred_value=clf.predict(myData[39, ].reshape(1,-1)) 

for j in range (50):   
   output[j]=clf.predict(test_data[j, ].reshape(1,-1))