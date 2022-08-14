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
import statistics


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
        
def ventana2():
    def datosEstadisticos(datos):
        lista_datos = []
        lista_datos.append(statistics.mean(datos))
        lista_datos.append(statistics.mode(datos))
        lista_datos.append(statistics.median(datos))
        lista_datos.append(stats.kurtosis(datos))
        lista_datos.append(stats.skew(datos))
        
        return lista_datos
    list_length=datosEstadisticos(datos['length'])
    list_Diameter=datosEstadisticos(datos['Diameter'])
    list_Height=datosEstadisticos(datos['Height'])
    list_Whole_weight=datosEstadisticos(datos['Whole weight'])
    list_Shucked_weight=datosEstadisticos(datos['Shucked weight'])
    list_Viscera_weight=datosEstadisticos(datos['Viscera weight'])
    list_Shell_weight=datosEstadisticos(datos['Shell weight'])
    list_Rings=datosEstadisticos(datos['Rings'])
    
    ventana_2 = tk.Tk()
    
    #titulo
    
    
    label_1 = tk.Label(ventana_2,text="Columm's Name",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_1.grid(padx=10, pady=10, row=1, column=0,)
    
    
    label_2 = tk.Label(ventana_2,text="length",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_2.grid(padx=10, pady=10, row=2, column=0, )
    
    label_3 = tk.Label(ventana_2,text="Diameter",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_3.grid(padx=10, pady=10, row=3, column=0, )
    
    label_4 = tk.Label(ventana_2,text="Height",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_4.grid(padx=10, pady=10, row=4, column=0, )
    
    label_5 = tk.Label(ventana_2,text="Whole weight",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_5.grid(padx=10, pady=10, row=5, column=0, )
    
    label_6 = tk.Label(ventana_2,text="Shucked weight",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_6.grid(padx=10, pady=10, row=6, column=0, )
    
    label_7 = tk.Label(ventana_2,text="Viscera weight",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_7.grid(padx=10, pady=10, row=7, column=0, )
    
    label_8 = tk.Label(ventana_2,text="Shell weight",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_8.grid(padx=10, pady=10, row=8, column=0, )
    
    label_9 = tk.Label(ventana_2,text="Rings",
                              fg="blue",
                              font="consolas 12 bold",borderwidth=3)
    label_9.grid(padx=10, pady=10, row=9, column=0, )
    
    
    
    label_10 = tk.Label(ventana_2,text="Mean",
                              fg="blue",
                              font="consolas 12 bold",)
    label_10.grid(padx=10, pady=10, row=1, column=1, )
    
    label_11 = tk.Label(ventana_2,text="mode",
                              fg="blue",
                              font="consolas 12 bold",)
    label_11.grid(padx=10, pady=10, row=1, column=2, )
    
    label_12 = tk.Label(ventana_2,text="median",
                              fg="blue",
                              font="consolas 12 bold",)
    label_12.grid(padx=10, pady=10, row=1, column=3, )
    
    label_13 = tk.Label(ventana_2,text="Kurtosis",
                              fg="blue",
                              font="consolas 12 bold",)
    label_13.grid(padx=10, pady=10, row=1, column=4, )
    
    label_14 = tk.Label(ventana_2,text="skewness",
                              fg="blue",
                              font="consolas 12 bold",)
    label_14.grid(padx=10, pady=10, row=1, column=5, )
    
    len_mean = tk.Label(ventana_2,text=str(list_length[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=2, column=1)
    len_mode = tk.Label(ventana_2,text=str(list_length[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=2, column=2)
    len_median = tk.Label(ventana_2,text=str(list_length[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=2, column=3)
    len_kurt = tk.Label(ventana_2,text=str(list_length[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=2, column=4)
    len_skewn = tk.Label(ventana_2,text=str(list_length[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=2, column=5)


    dia_mean = tk.Label(ventana_2,text=str(list_Diameter[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=3, column=1)
    dia_mode = tk.Label(ventana_2,text=str(list_Diameter[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=3, column=2)
    dia_median = tk.Label(ventana_2,text=str(list_Diameter[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=3, column=3)
    dia_kurt = tk.Label(ventana_2,text=str(list_Diameter[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=3, column=4)
    dia_skewn = tk.Label(ventana_2,text=str(list_Diameter[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=3, column=5)

    hei_mean = tk.Label(ventana_2,text=str(list_Height[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=4, column=1)
    hei_mode = tk.Label(ventana_2,text=str(list_Height[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=4, column=2)
    hei_median = tk.Label(ventana_2,text=str(list_Height[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=4, column=3)
    hei_kurt = tk.Label(ventana_2,text=str(list_Height[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=4, column=4)
    hei_skewn = tk.Label(ventana_2,text=str(list_Height[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=4, column=5)

    whole_mean = tk.Label(ventana_2,text=str(list_Whole_weight[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=5, column=1)
    whole_mode = tk.Label(ventana_2,text=str(list_Whole_weight[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=5, column=2)
    whole_median = tk.Label(ventana_2,text=str(list_Whole_weight[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=5, column=3)
    whole_kurt = tk.Label(ventana_2,text=str(list_Whole_weight[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=5, column=4)
    whole_skewn = tk.Label(ventana_2,text=str(list_Whole_weight[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=5, column=5)

    shuck_mean = tk.Label(ventana_2,text=str(list_Shucked_weight[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=6, column=1)
    shuck_mode = tk.Label(ventana_2,text=str(list_Shucked_weight[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=6, column=2)
    shuck_median = tk.Label(ventana_2,text=str(list_Shucked_weight[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=6, column=3)
    shuck_kurt = tk.Label(ventana_2,text=str(list_Shucked_weight[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=6, column=4)
    shuck_skewn = tk.Label(ventana_2,text=str(list_Shucked_weight[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=6, column=5)

    vis_mean = tk.Label(ventana_2,text=str(list_Viscera_weight[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=7, column=1)
    vis_mode = tk.Label(ventana_2,text=str(list_Viscera_weight[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=7, column=2)
    vis_median = tk.Label(ventana_2,text=str(list_Viscera_weight[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=7, column=3)
    vis_kurt = tk.Label(ventana_2,text=str(list_Viscera_weight[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=7, column=4)
    vis_skewn = tk.Label(ventana_2,text=str(list_Viscera_weight[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=7, column=5)

    she_mean = tk.Label(ventana_2,text=str(list_Shell_weight[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=8, column=1)
    she_mode = tk.Label(ventana_2,text=str(list_Shell_weight[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=8, column=2)
    she_median = tk.Label(ventana_2,text=str(list_Shell_weight[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=8, column=3)
    she_kurt = tk.Label(ventana_2,text=str(list_Shell_weight[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=8, column=4)
    she_skewn = tk.Label(ventana_2,text=str(list_Shell_weight[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=8, column=5)

    rin_mean = tk.Label(ventana_2,text=str(list_Rings[0]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=9, column=1)
    rin_mode = tk.Label(ventana_2,text=str(list_Rings[1]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=9, column=2)
    rin_median = tk.Label(ventana_2,text=str(list_Rings[2]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=9, column=3)
    rin_kurt = tk.Label(ventana_2,text=str(list_Rings[3]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=9, column=4)    
    rin_skewn = tk.Label(ventana_2,text=str(list_Rings[4]),bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=10, pady=10, row=9, column=5)


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
print(stats.kurtosis(datos['Diameter']))



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

my_button_2 = tk.Button(ventana, text="DataSet Info", font="consolas 14 bold", command=ventana2)
my_button_2.grid(padx=10, pady=10, row=4, column=4, columnspan=2)


ventana.mainloop()








   

    
    













