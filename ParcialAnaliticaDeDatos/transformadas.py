# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:44:06 2022

@author: johan
"""

# Import libraries 
import statistics
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#X=new_Data_2[['PT08.S1(CO)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S5(O3)']]
datos = pd.read_excel('new_Data_2.xlsx')
i='PT08.S5(O3)'
X=datos[i]
elevado = pow(np.array(X).reshape(-1,1), 2)
raiz = pow(np.array(X).reshape(-1,1), 0.5)
exponencial = np.exp(np.array(X).reshape(-1,1))
logaritmo = np.log(np.array(X).reshape(-1,1))
t = "Raiz: ",i
t2 = "Elevado: ",i
t3 = "exponencial: ", i
t4 = "logaritmo", i
plot.hist(raiz,bins=20)
plot.title(t)
plot.subplots()
plot.hist(elevado,bins=20)
plot.title(t2)
plot.subplots()    
plot.hist(exponencial,bins=20)
plot.title(t3)
plot.subplots()
plot.hist(logaritmo,bins=20)
plot.title(t4)
