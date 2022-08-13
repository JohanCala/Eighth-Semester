# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 09:54:48 2022

@author: johan
"""

### paquetes de impotacion y manejo de datos
import pandas as pd
import numpy as np

### paquetes de graficos
import matplotlib.pyplot as plot
from scipy import stats



######  paquetes de analitica de datos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#####   paquetes de interfaz grafica
	
import tkinter as tk
from tkinter import ttk


### Funciones
def IdentificarAtipicos(df, ft, valorAlfa):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1
    
    bigote_inferior = q1 - valorAlfa * iqr
    bigote_superior = q3 + valorAlfa * iqr
    
    ls = df.index[(df[ft]<bigote_inferior) | (df[ft] > bigote_superior)]
    return ls

def eliminar(df, index):
    index = sorted(set(index))
    df = df.drop(index)
    return df

def Histograma(df,col):
    plot.title(col + "----Histograma")
    plot.hist(df[col])
    plot.show()


def Bigotes(df, col):
    plot.title(col + "----Diagrama de Caja")
    plot.boxplot(df[col])
    plot.show()

def Normalización(df, col):
    fig=plot.figure()
    ax=fig.add_subplot(111)
    res=stats.probplot(df[col],dist=stats.norm,plot=ax)
    plot.show()

def Dispersión(df,col1,col2):
    plot.scatter((df[col1]), (df[col2]))
    plot.xlabel(col1)
    plot.ylabel(col2)
    plot.show()

def Correlación(df,col1,col2):
    plot.xcorr(df[col1], df[col2])
    plot.xlabel(col1)
    plot.ylabel(col2)
    plot.show()

def Graph():
    dataframe = datos   
    atipico = com_graph_entrada_3.get()
    if atipico == 'Yes':
        valor_alpha = float(entrada_num_1.get())
        index_list = []
        for i in aux:
            index_list.extend(IdentificarAtipicos(dataframe, i,valor_alpha))
        final_index_list = []
        for index in index_list:
            if index not in final_index_list:
                final_index_list.append(index)

        dataframe = eliminar(dataframe,index_list)
        dataframe.columns=columnas
    else:
        dataframe = datos
    
    type_plot = com_graph_entrada.get()
    name_col = desplegable_entrada.get()
    
    
    
    if type_plot == 'Histogram':
        Histograma(dataframe, name_col)
    elif type_plot == 'BoxPlot':
        Bigotes(dataframe, name_col)
    elif type_plot == 'Normalization':
        Normalización(dataframe,name_col)
    elif type_plot == 'Scatter':
        name_col_2 = desplegable_entrada_2.get()
        Dispersión(dataframe, name_col, name_col_2)
    else:
        tk.messagebox.showinfo(message="please select all the options ", title="Alert")
    
    com_graph_entrada.set("")
    desplegable_entrada.set("")
    desplegable_entrada_2.set("")
    desplegable_entrada_2["state"]="disable"

def validacion(event):
    if com_graph_entrada.get() == 'Scatter':
        
            
            new_aux = aux
            new_aux.remove(desplegable_entrada.get())
            desplegable_entrada_2["values"]=new_aux
            desplegable_entrada_2["state"]="readonly"
        
        
    else:
        desplegable_entrada_2["state"]="disabled"
        
def validacion_2(event):
    if com_graph_entrada_3.get() == "Yes":
        entrada_num_1['state'] = "normal"
    else:
        entrada_num_1['state'] = "disabled"
#### Main

archivo='abalone.csv'

 
datos=pd.read_csv(archivo)
columnas=['sex', 'length',
'Diameter',
'Height',
'Whole weight',
'Shucked weight',
'Viscera weight',
'Shell weight',
'Rings' ]
datos.columns=columnas

 
aux = ['length',
'Diameter',
'Height',
'Whole weight',
'Shucked weight',
'Viscera weight',
'Shell weight',
'Rings' ]
index_list = []
for i in aux:
    index_list.extend(IdentificarAtipicos(datos, i,1.5))
final_index_list = []
for index in index_list:
    if index not in final_index_list:
        final_index_list.append(index)

new_df = eliminar(datos,index_list)
new_df.columns=columnas


### modelo de una sola entrada

X=datos['Diameter']
Y=datos['Rings']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        Y,
                                        train_size   = 0.5,
                                    )

modelo = LinearRegression()
modelo.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
print('entrada longitud, salida anillos',modelo.score(np.array(X).reshape(-1, 1), Y))

predicciones = modelo.predict(X = np.array(X_test).reshape(-1,1))
rmse = mean_squared_error(y_true  = y_test, y_pred  = predicciones)
print('el valor del rmse es',rmse)



#### ventana

ventana = tk.Tk()
ventana.title("a")


ventana.configure(bg="lightblue")

#labels



titulo = tk.Label(ventana,text='Abalone Analysis DataSet',
                          bg="lightblue",fg="black",
                          font="consolas 20 bold")
titulo.grid(padx=20, pady=20, row=0, column=0, columnspan=2)

rotulo_combo_3 = tk.Label(ventana,text='outliers: ',
                          bg="lightblue",fg="black",
                          font="consolas 14 bold")
rotulo_combo_3.grid(padx=10, pady=10, row=1, column=0, )

rotulo_combo_4 = tk.Label(ventana,text='Type your alpha value: ',
                          bg="lightblue",fg="black",
                          font="consolas 14 bold")
rotulo_combo_4.grid(padx=10, pady=10, row=1, column=2, )
rotulo_combo_2 = tk.Label(ventana,text='Select your chart type: ',
                          bg="lightblue",fg="black",
                          font="consolas 14 bold")
rotulo_combo_2.grid(padx=10, pady=10, row=3, column=0, )

rotulo_combo_1 = tk.Label(ventana,text='Select your variable: ',
                          bg="lightblue",fg="black",
                          font="consolas 14 bold")
rotulo_combo_1.grid(padx=10, pady=10, row=2, column=0, )

#entradadetexto

entrada_num_1 = tk.Entry(ventana,bg="White",fg="black",font="consolas 14 bold",state="disabled")
entrada_num_1.grid(padx=10, pady=10, row=1, column=4)

#ComboBox
desplegable_entrada = ttk.Combobox(ventana,font="consolas 14 bold",
                                  width=16,
                                  values=aux,
                                  state="readonly")
desplegable_entrada.grid(padx=10, pady=10, row=2, column=1, )
desplegable_entrada.set('')

desplegable_entrada_2 = ttk.Combobox(ventana,font="consolas 14 bold",
                                  width=16,
                                  values=aux,
                                  state="disabled")
desplegable_entrada_2.grid(padx=10, pady=10, row=2, column=2,)
desplegable_entrada_2.set('')




com_graph_entrada = ttk.Combobox(ventana,font="consolas 14 bold",
                                  width=16,
                                  values=['Histogram','BoxPlot','Normalization','Scatter'],
                                  state="readonly")
com_graph_entrada.grid(padx=10, pady=10, row=3, column=1)
com_graph_entrada.set('')
com_graph_entrada.bind("<<ComboboxSelected>>", validacion)

com_graph_entrada_3 = ttk.Combobox(ventana,font="consolas 14 bold",
                                  width=16,
                                  values=['Yes','No'],
                                  state="readonly")
com_graph_entrada_3.grid(padx=10, pady=10, row=1, column=1)
com_graph_entrada_3.set('')
com_graph_entrada_3.bind("<<ComboboxSelected>>", validacion_2)


#Botones


my_button = tk.Button(ventana, text="Graph It!", font="consolas 14 bold", command=Graph)
my_button.grid(padx=10, pady=10, row=4, column=0, columnspan=2)

ventana.mainloop()




   

    
    













