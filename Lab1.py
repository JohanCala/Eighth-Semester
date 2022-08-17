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
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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



def Graph():
    dataframe = datos.copy()   
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
        dataframe = datos.copy()
    
    type_plot = com_graph_entrada.get()
    name_col = desplegable_entrada.get()
    
    ventana_graf = tk.Tk()
    ventana_graf.title("Data analytics and artificial intelligence")
   
    
    fig = Figure(figsize=(5, 4), dpi=100)
    plot=fig.add_subplot(111)

    
    if type_plot == 'Histogram':
        #plot.title(col + "----Histograma")
        plot.hist(dataframe[name_col])
        
    elif type_plot == 'BoxPlot':
        
        plot.boxplot(dataframe[name_col])
        

    elif type_plot == 'Normalization':
       
       ax=fig.add_subplot(111)
       res=stats.probplot(dataframe[name_col],dist=stats.norm,plot=ax)
       
    elif type_plot == 'Scatter':
        name_col_2 = desplegable_entrada_2.get()
        plot.scatter((dataframe[name_col]), (dataframe[name_col_2]))
    else:
        tk.messagebox.showinfo(message="Please select all the options ", title="Alert")
        
    canvas = FigureCanvasTkAgg(fig, master=ventana_graf)
    canvas.draw()
    canvas.get_tk_widget().pack() 
    canvas.get_tk_widget().pack()
    
    com_graph_entrada.set("")
    desplegable_entrada.set("")
    desplegable_entrada_2.set("")
    desplegable_entrada_2["state"]="disable"
    entrada_num_1["text"]=""

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
    ventana_2.title("Data analytics and artificial intelligence")
    
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


def Ventana3():
    
    def Calcular():
        if desplegable_entrada_Y.get() == desplegable_entrada_X.get():
            
            tk.messagebox.showinfo(message="Please do not be useless, select different values for X and Y ", title="Alert")
        else:
            dataframe = datos.copy() 
            valor_alpha = float(entrada_num_1.get())
            
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
            ##### Con atipicos
            X=datos[desplegable_entrada_X]
            Y=datos[desplegable_entrada_Y]

            X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    Y,
                                                    train_size   = 0.5,
                                                )

            modelo = LinearRegression()
            modelo.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
            r_score=modelo.score(np.array(X).reshape(-1, 1), Y)

            predicciones = modelo.predict(X = np.array(X_test).reshape(-1,1))
            rmse = mean_squared_error(y_true  = y_test, y_pred  = predicciones)
            
            label_model_Rmse["text"] = str(rmse)
            label_model_score["text"] = str(r_score)
            
            #### Sin atipicos
            
            X=dataframe[desplegable_entrada_X]
            Y=dataframe[desplegable_entrada_Y]

            X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    Y,
                                                    train_size   = 0.5,
                                                )

            modelo_sin = LinearRegression()
            modelo_sin.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
            r_score_sin_a=modelo_sin.score(np.array(X).reshape(-1, 1), Y)

            predicciones_sin = modelo_sin.predict(X = np.array(X_test).reshape(-1,1))
            rmse_sin = mean_squared_error(y_true  = y_test, y_pred  = predicciones_sin)
            
            
            label_model_Rmse_sin_a["text"]=str(rmse_sin)
            label_model_score_sin_A["text"]=str(r_score_sin_a)
            
    ventana = tk.Tk()
    ventana.title("Data analytics")
    ventana.configure(bg="lightblue")
    titulo = tk.Label(ventana,text='Linear Regression Model Comparison',bg="lightblue",
                              fg="blue",
                              font="consolas 25 bold")
    titulo.grid(padx=20, pady=20, row=0, column=0, columnspan=4)
    subt = tk.Label(ventana,text='Type your alpha value:',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold").grid(padx=20, pady=20, row=1, column=0)
    entrada_num_1 = tk.Entry(ventana,bg="White",fg="black",font="consolas 14 bold",state="normal").grid(padx=5, pady=5, row=1, column=1)

    #una sola entrada 
    label_1 = tk.Label(ventana,text='One input',
                              bg="lightblue",fg="black",
                              font="consolas 18 bold").grid(padx=5, pady=5, row=2, column=0,columnspan=2)
    label_2 = tk.Label(ventana,text='Select X:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=3, column=0)

    desplegable_entrada_X = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly").grid(padx=5, pady=5, row=3, column=1,)

    label_3 = tk.Label(ventana,text='Select Y:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=4, column=0)
    desplegable_entrada_Y = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly").grid(padx=5, pady=5, row=4, column=1,)

    my_button_1 = tk.Button(ventana, text="Predict", font="consolas 14 bold", command=Calcular).grid(padx=5, pady=5, row=5, column=0,columnspan=2 )
    label_8 = tk.Label(ventana,text='With the outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=6, column=0,columnspan=2)
    label_4 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=7, column=0)

    label_model_score = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=7, column=1)

    label_5 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=8, column=0)

    label_model_Rmse = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=8, column=1)
    label_9 = tk.Label(ventana,text='Without outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=9, column=0,columnspan=2)


    label_6 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=0)

    label_model_score_sin_A = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=1)

    label_7 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=11, column=0)

    label_model_Rmse_sin_a = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=11, column=1)

    label_aux = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=2, column=4)



    ########################################## entradas multiples

    label_10 = tk.Label(ventana,text='Multiple Entries',
                              bg="lightblue",fg="black",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=0, column=5)

    label_11 = tk.Label(ventana,text='Select X:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=1, column=5,columnspan=2)


    tk.Checkbutton(ventana,text="Lenght        ",bg="lightblue",font="consolas 10 bold",variable=aux[0]).grid(padx=5, pady=5, row=2, column=5)
    tk.Checkbutton(ventana,text="Diameter      ",bg="lightblue",font="consolas 10 bold",variable=aux[1]).grid(padx=5, pady=5, row=3, column=5)
    tk.Checkbutton(ventana,text="Whole weight  ",bg="lightblue",font="consolas 10 bold",variable=aux[2]).grid(padx=5, pady=5, row=4, column=5)
    tk.Checkbutton(ventana,text="Whole weight  ",bg="lightblue",font="consolas 10 bold",variable=aux[3]).grid(padx=5, pady=5, row=5, column=5)
    tk.Checkbutton(ventana,text="Shucked weight",bg="lightblue",font="consolas 10 bold",variable=aux[4]).grid(padx=5, pady=5, row=2, column=6)
    tk.Checkbutton(ventana,text="Viscera weight",bg="lightblue",font="consolas 10 bold",variable=aux[5]).grid(padx=5, pady=5, row=3, column=6)
    tk.Checkbutton(ventana,text="Shell weight  ",bg="lightblue",font="consolas 10 bold",variable=aux[6]).grid(padx=5, pady=5, row=4, column=6)
    tk.Checkbutton(ventana,text="Rings         ",bg="lightblue",font="consolas 10 bold",variable=aux[7]).grid(padx=5, pady=5, row=5, column=6)

    label_12 = tk.Label(ventana,text='Select Y:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=6, column=5)
    desplegable_entrada_Y = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly").grid(padx=5, pady=5, row=6, column=6)

    my_button_1 = tk.Button(ventana, text="Predict", font="consolas 14 bold", command=Calcular).grid(padx=5, pady=5, row=7, column=5,columnspan=2 )

    label_13 = tk.Label(ventana,text='With the outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=8, column=5,columnspan=2)


    label_15 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=9, column=5)

    label_model_score_mul = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=9, column=6)

    label_16 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=5)

    label_model_Rmse_mul = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=6)

    label_14 = tk.Label(ventana,text='Without outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=11, column=5,columnspan=2)

    label_17 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=12, column=5)

    label_model_score_sin_A = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=12, column=6)

    label_18 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=13, column=5)

    label_model_Rmse_sin_a = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=13, column=6)





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

X=datos['length']
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
ventana.title("Data analytics and artificial intelligence")


ventana.configure(bg="lightblue")

#labels



titulo = tk.Label(ventana,text='Abalone Analysis DataSet',
                          bg="lightblue",fg="black",
                          font="consolas 20 bold")
titulo.grid(padx=20, pady=20, row=0, column=0, columnspan=5)

rotulo_combo_3 = tk.Label(ventana,text='Outliers: ',
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

my_button_2 = tk.Button(ventana, text="DataSet Info!", font="consolas 14 bold", command=ventana2)
my_button_2.grid(padx=10, pady=10, row=4, column=2, columnspan=2)

my_button_3 = tk.Button(ventana, text="Regression Model!", font="consolas 14 bold", command=Ventana3)
my_button_3.grid(padx=10, pady=10, row=4, column=4, columnspan=2)


ventana.mainloop()








   

    
    













