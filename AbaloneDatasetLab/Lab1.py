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
import math 


######  paquetes de analitica de datos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#####   paquetes de interfaz grafica
	
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox



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
    
    
    




def ventana3():
    def clean():
        entrada_num_1.delete(0,"end")
        label_model_Rmse_con_atip['text'] = ""
        label_model_score_con_atip['text'] = ""
        label_model_Rmse_sin_atip['text'] = ""
        label_model_score_sin_atip['text'] = ""
        
    def clean2():
        entrada_num_1.delete(0,"end")
        desplegable_entrada_Y_m.set("")
        regre_multi_con_a_rmse['text'] = ""
        regre_multi_con_a_scor['text'] = ""
        
        regre_multi_sin_a_rmse['text'] = ""
        regre_multi_sin_a_scor['text'] = ""
        
        
    def CalcularM():
        if entrada_num_1.get() == "":
            tk.messagebox.showinfo(message="please enter the value of alpha ", title="Alert")
        x = []
        if seleccion1.get()==1:
            x.append(check1['text'].strip())
        if seleccion2.get()==1:
            x.append(check2['text'].strip())
        if seleccion3.get()==1:
            x.append(check3['text'].strip())
        if seleccion4.get()==1:
            x.append(check4['text'].strip())
        if seleccion5.get()==1:
            x.append(check5['text'].strip())
        if seleccion6.get()==1:
            x.append(check6['text'].strip())
        if seleccion7.get()==1:
            x.append(check7['text'].strip())
        if seleccion8.get()==1:
            x.append(check8['text'].strip())
        
        
        
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
        
        #Con atipicos 
        X=datos[x]
        Y=datos[desplegable_entrada_Y_m.get()]
        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)

        model.fit(X = X_train, y = y_train)

        y_predict = model.predict(X = np.array(X_test))

        r2_m_con = r2_score(y_true=y_test,y_pred=y_predict)
        rmse_m_con = mean_squared_error(y_true=y_test, y_pred=y_predict)
        
        regre_multi_con_a_rmse['text'] = str(rmse_m_con)
        regre_multi_con_a_scor['text'] = str(r2_m_con)
        #Sin atipicos
        X=dataframe[x]
        Y=dataframe[desplegable_entrada_Y_m.get()]
        
        model_a = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)

        model_a.fit(X = X_train, y = y_train)

        y_predict = model_a.predict(X = np.array(X_test))

        r2_m_sin_a = r2_score(y_true=y_test,y_pred=y_predict)
        rmse_sin_a = mean_squared_error(y_true=y_test, y_pred=y_predict)
        
        regre_multi_con_a_rmse['text'] = str(rmse_m_con)
        regre_multi_con_a_scor['text'] = str(r2_m_con)
        
        regre_multi_sin_a_rmse['text'] = str(rmse_sin_a)
        regre_multi_sin_a_scor['text'] = str(r2_m_sin_a)
        
        

    def Calcular():
        if entrada_num_1.get() == "":
            tk.messagebox.showinfo(message="please enter the value of alpha ", title="Alert")
            
            
        if desplegable_entrada_X.get() == desplegable_entrada_Y.get():
            
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
            X=datos[desplegable_entrada_X.get()]
            Y=datos[desplegable_entrada_Y.get()]

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
            
            
            
            
            
            
            
            
            
            
            #### Sin atipicos
            
            X=dataframe[desplegable_entrada_X.get()]
            Y=dataframe[desplegable_entrada_Y.get()]

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
            
            label_model_Rmse_con_atip['text'] = str(rmse)
            label_model_score_con_atip['text'] = str(r_score)
            label_model_Rmse_sin_atip['text'] = str(rmse_sin)
            label_model_score_sin_atip['text'] = str(r_score_sin_a)
            
            
            
            
    aux = ['length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Viscera weight',
    'Shell weight',
    'Rings' ]

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
    entrada_num_1 = tk.Entry(ventana,bg="White",fg="black",font="consolas 14 bold",state="normal")
    entrada_num_1.grid(padx=5, pady=5, row=1, column=1)
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
                                      state="readonly")
    desplegable_entrada_X.grid(row=3, column=1)


    label_3 = tk.Label(ventana,text='Select Y:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=4, column=0)
    desplegable_entrada_Y = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_entrada_Y.grid(row=4, column=1)


    my_button_1 = tk.Button(ventana, text="Predict", font="consolas 14 bold", command=Calcular).grid(padx=5, pady=5, row=5, column=0,columnspan=2 )
    label_8 = tk.Label(ventana,text='With the outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=6, column=0,columnspan=2)
    label_4 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=7, column=0)



    label_5 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=8, column=0)




    label_9 = tk.Label(ventana,text='Without outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=9, column=0,columnspan=2)


    label_6 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=0)



    label_7 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=11, column=0)




    my_button_1 = tk.Button(ventana, text="Clean", font="consolas 14 bold", command=clean).grid(padx=5, pady=5, row=12, column=0,columnspan=2 )


    label_aux = tk.Label(ventana,text='',
                              bg="lightblue",fg="black",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=2, column=4)

    label_model_Rmse_con_atip = tk.Label(ventana, text="",bg="white",fg="black",font="consolas 14 bold")
    label_model_score_con_atip = tk.Label(ventana, text="",bg="white",fg="black",font="consolas 14 bold")
    label_model_Rmse_sin_atip = tk.Label(ventana, text="",bg="white",fg="black",font="consolas 14 bold")
    label_model_score_sin_atip= tk.Label(ventana, text="",bg="white",fg="black",font="consolas 14 bold")

    label_model_Rmse_con_atip.grid(padx=5, pady=5, row=8, column=1)
    label_model_score_con_atip.grid(padx=5, pady=5, row=7, column=1)
    label_model_Rmse_sin_atip.grid(padx=5, pady=5, row=11, column=1)
    label_model_score_sin_atip.grid(padx=5, pady=5, row=10, column=1)



    ########################################## entradas multiples

    label_10 = tk.Label(ventana,text='Multiple Entries',
                              bg="lightblue",fg="black",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=0, column=5)

    label_11 = tk.Label(ventana,text='Select X:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=1, column=5,columnspan=2)

    seleccion1=tk.IntVar()

    seleccion2=tk.IntVar()

    seleccion3=tk.IntVar()

    seleccion4=tk.IntVar()

    seleccion5=tk.IntVar()

    seleccion6=tk.IntVar()

    seleccion7=tk.IntVar()

    seleccion8=tk.IntVar()

    check1 = tk.Checkbutton(ventana,text="length        ",bg="lightblue",font="consolas 10 bold",variable=seleccion1)
    check2 = tk.Checkbutton(ventana,text="Diameter      ",bg="lightblue",font="consolas 10 bold",variable=seleccion2)
    check3 = tk.Checkbutton(ventana,text="Height        ",bg="lightblue",font="consolas 10 bold",variable=seleccion3)
    check4 = tk.Checkbutton(ventana,text="Whole weight  ",bg="lightblue",font="consolas 10 bold",variable=seleccion4)
    check5 = tk.Checkbutton(ventana,text="Shucked weight",bg="lightblue",font="consolas 10 bold",variable=seleccion5)
    check6 = tk.Checkbutton(ventana,text="Viscera weight",bg="lightblue",font="consolas 10 bold",variable=seleccion6)
    check7 = tk.Checkbutton(ventana,text="Shell weight  ",bg="lightblue",font="consolas 10 bold",variable=seleccion7)
    check8 = tk.Checkbutton(ventana,text="Rings         ",bg="lightblue",font="consolas 10 bold",variable=seleccion8)

    check1.grid(padx=5, pady=5, row=2, column=5,)
    check2.grid(padx=5, pady=5, row=3, column=5)
    check3.grid(padx=5, pady=5, row=4, column=5)
    check4.grid(padx=5, pady=5, row=5, column=5)
    check5.grid(padx=5, pady=5, row=2, column=6)
    check6.grid(padx=5, pady=5, row=3, column=6)
    check7.grid(padx=5, pady=5, row=4, column=6)
    check8.grid(padx=5, pady=5, row=5, column=6)

    label_12 = tk.Label(ventana,text='Select Y:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=6, column=5)
    desplegable_entrada_Y_m = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_entrada_Y_m.grid(padx=5, pady=5, row=6, column=6)

    my_button_2 = tk.Button(ventana, text="Predict", font="consolas 14 bold", command=CalcularM).grid(padx=5, pady=5, row=7, column=5,columnspan=2 )

    label_13 = tk.Label(ventana,text='With the outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=8, column=5,columnspan=2)


    label_15 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=9, column=5)



    label_16 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=10, column=5)



    label_14 = tk.Label(ventana,text='Without outliers',
                              bg="lightblue",fg="blue",
                              font="consolas 20 bold").grid(padx=5, pady=5, row=11, column=5,columnspan=2)

    label_15 = tk.Label(ventana,text='Model Score:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=12, column=5)



    label_16 = tk.Label(ventana,text='RMSE value:',
                              bg="lightblue",fg="black",
                              font="consolas 14 bold").grid(padx=5, pady=5, row=13, column=5)

    regre_multi_sin_a_rmse = tk.Label(ventana,text='',
                              bg="white",fg="black",
                              font="consolas 14 bold")


    regre_multi_sin_a_scor = tk.Label(ventana,text='',
                              bg="white",fg="black",
                              font="consolas 14 bold")

    regre_multi_con_a_rmse = tk.Label(ventana,text='',
                              bg="white",fg="black",
                              font="consolas 14 bold")

    regre_multi_con_a_scor = tk.Label(ventana,text='',
                              bg="white",fg="black",
                              font="consolas 14 bold")

    regre_multi_con_a_rmse.grid(padx=5, pady=5, row=10, column=6)
    regre_multi_con_a_scor.grid(padx=5, pady=5, row=9, column=6)

    regre_multi_sin_a_rmse.grid(padx=5, pady=5, row=13, column=6)
    regre_multi_sin_a_scor.grid(padx=5, pady=5, row=12, column=6)

    my_button_2 = tk.Button(ventana, text="Clean", font="consolas 14 bold", command=clean2).grid(padx=5, pady=5, row=14, column=5,columnspan=2 )

def ventana4():
    def cleardata():
        datos = datosbk
    def modeloTransformadas():   #'none','exponential','square root','squared'
        if desplegable_t_1.get() == 'exponential':
            print(":D")  
        elif desplegable_t_1.get() == 'square root':
            datos['length'] = pow(datos['length'], (1/2))
        elif desplegable_t_1.get() == 'squared':
            datos['length'] = pow(datos['length'], (2))
        else:
            datos['length'] = datos['length']
            
        if desplegable_t_2.get() == 'exponential':
            print(":D")  
        elif desplegable_t_2.get() == 'square root':
            datos['Diameter'] = pow(datos['Diameter'], (1/2))
        elif desplegable_t_2.get() == 'squared':
            datos['Diameter'] = pow(datos['Diameter'], (2))
        else:
            datos['Diameter'] = datos['Diameter']
            
        if desplegable_t_3.get() == 'exponential':
            print(":D")  
        elif desplegable_t_3.get() == 'square root':
            datos['Height'] = pow(datos['Height'], (1/2))
        elif desplegable_t_3.get() == 'squared':
            datos['Height'] = pow(datos['Height'], (2))
        else:
            datos['Height'] = datos['Height']
            
            
        if desplegable_t_4.get() == 'exponential':
            print(":D")  
        elif desplegable_t_4.get() == 'square root':
            datos['Whole weight'] = pow(datos['Whole weight'], (1/2))
        elif desplegable_t_4.get() == 'squared':
            datos['Whole weight'] = pow(datos['Whole weight'], (2))
        else:
            datos['Whole weight'] = datos['Whole weight']
            
        if desplegable_t_5.get() == 'exponential':
            print(":D")  
        elif desplegable_t_5.get() == 'square root':
            datos['Shucked weight'] = pow(datos['Shucked weight'], (1/2))
        elif desplegable_t_5.get() == 'squared':
            datos['Shucked weight'] = pow(datos['Shucked weight'], (2))
        else:
            datos['Shucked weight'] = datos['Shucked weight']
            
        if desplegable_t_6.get() == 'exponential':
            print(":D")  
        elif desplegable_t_6.get() == 'square root':
            datos['Viscera weight'] = pow(datos['Viscera weight'], (1/2))
        elif desplegable_t_6.get() == 'squared':
            datos['Viscera weight'] = pow(datos['Viscera weight'], (2))
        else:
            datos['Viscera weight'] = datos['Viscera weight']
        if desplegable_t_7.get() == 'exponential':
            print(":D")  
        elif desplegable_t_7.get() == 'square root':
            datos['Shell weight'] = pow(datos['Shell weight'], (1/2))
        elif desplegable_t_7.get() == 'squared':
            datos['Shell weight'] = pow(datos['Shell weight'], (2))
        else:
            datos['Shell weight'] = datos['Shell weight']
            
        
        X=datos[['length',
        'Diameter',
        'Height',
        'Whole weight',
        'Shucked weight',
        'Viscera weight',
        'Shell weight']]
        Y=datos['Rings']
        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(
                                                X,
                                                Y,
                                                train_size   = 0.5,
                                            )

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
        model.fit(X = X_train, y = y_train)
        y_predict = model.predict(X = np.array(X_test))
        r2 = r2_score(y_true=y_test,y_pred=y_predict)
        messagebox.showinfo(message=r2, title="ScoreModel")


    aux = ['none','exponential','square root','squared']

    ventana = tk.Tk()
    ventana.title("Data analytics and artificial intelligence")
    ventana.configure(bg="lightblue")

    titulo = tk.Label(ventana,text='Transformations',bg="lightblue",
                              fg="blue",
                              font="consolas 25 bold")
    titulo.grid(padx=5, pady=5, row=0, column=0, columnspan=4)

    #colums labels

    label_1 = tk.Label(ventana,text='length',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=1, column=0)

    label_1 = tk.Label(ventana,text='Diameter',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=2, column=0)

    label_1 = tk.Label(ventana,text='Height',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=3, column=0)

    label_1 = tk.Label(ventana,text='Whole weight',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=4, column=0)

    label_1 = tk.Label(ventana,text='Shucked weight',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=5, column=0)

    label_1 = tk.Label(ventana,text='Viscera weight',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=6, column=0)

    label_1 = tk.Label(ventana,text='Shell weight',bg="lightblue",
                              fg="black",
                              font="consolas 14 bold")
    label_1.grid(padx=1, pady=1, row=7, column=0)


    #Combo Box

    desplegable_t_1 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_1.grid(padx=1, pady=1, row=1, column=1)

    desplegable_t_2 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_2.grid(padx=1, pady=1, row=2, column=1)

    desplegable_t_3 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_3.grid(padx=1, pady=1, row=3, column=1)

    desplegable_t_4 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_4.grid(padx=1, pady=1, row=4, column=1)

    desplegable_t_5 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_5.grid(padx=1, pady=1, row=5, column=1)

    desplegable_t_6 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_6.grid(padx=1, pady=1, row=6, column=1)

    desplegable_t_7 = ttk.Combobox(ventana,font="consolas 14 bold",
                                      width=16,
                                      values=aux,
                                      state="readonly")
    desplegable_t_7.grid(padx=1, pady=1, row=7, column=1)


    #botones
    my_button_t_1 = tk.Button(ventana, text="Score", font="consolas 14 bold", command=modeloTransformadas)
    my_button_t_1.grid(padx=1, pady=1, row=8, column=0 )

    my_button_t_2 = tk.Button(ventana, text="Clear", font="consolas 14 bold", command=cleardata)
    my_button_t_2.grid(padx=1, pady=1, row=8, column=1)

    #Archivo




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
    datosbk = datos.copy()


        





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

##"# modelo de una "sola entrada

"""X=datos['length']
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
print('el valor del rmse es',rmse)"""




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

my_button_3 = tk.Button(ventana, text="Regression Model!", font="consolas 14 bold",command=ventana3)
my_button_3.grid(padx=10, pady=10, row=4, column=4, columnspan=2)

my_button_4 = tk.Button(ventana, text="Transformations", font="consolas 14 bold",command=ventana4)
my_button_4.grid(padx=10, pady=10, row=3, column=3, columnspan=2)


ventana.mainloop()








   

    
    













