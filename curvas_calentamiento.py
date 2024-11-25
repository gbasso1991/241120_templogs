#%% Templogs / Transiciones de Fase / Calores especificos
'''
Analizo templogs
'''
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy
from datetime import datetime,timedelta
import matplotlib as mpl
from scipy.interpolate import CubicSpline,PchipInterpolator

#%%
def lector_templog(directorio,rango_T_fijo=True):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt).
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura y plotea el log completo
    '''
    data = pd.read_csv(directorio,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python')

    temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2= pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])
    return timestamp,temp_CH1, temp_CH2

def procesar_temperatura(directorio,rango_T_fijo=True):
    # Obtener archivos de datos y archivos de templog
    paths_m = glob(os.path.join(directorio, '*.txt'))
    paths_m.sort()
    paths_T = glob(os.path.join(directorio, '*templog*'))
    
    # Levantar fechas de archivos grabadas en meta
    Fechas = []
    for fp in paths_m:
        with open(fp, 'r') as f:
            fecha_in_file = f.readline()
            Fechas.append(fecha_in_file.split()[-1])
    
    # Obtener timestamps y temperaturas del templog
    timestamp, temperatura, __ = lector_templog(paths_T[0])
    
    # Calcular tiempos completos en segundos
    t_full = np.array([(t - timestamp[0]).total_seconds() for t in timestamp])
    T_full = temperatura

    # Procesar las fechas y tiempos de los archivos
    dates = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas[:-1]]  # datetimes con fecha de archivos
    time_delta = [t.total_seconds() for t in np.diff(dates)]  # diferencia de tiempo entre archivos
    time_delta.insert(0, 0)  # Insertar el primer delta como 0
    delta_0 = (dates[0] - timestamp[0]).total_seconds()  # diferencia entre comienzo templog y 1er archivo

    # Buscar los índices de los datos de templog correspondientes al primer y último archivo
    indx_1er_dato = np.nonzero(timestamp == dates[0].replace(microsecond=0))[0][0]
    indx_ultimo_dato = np.nonzero(timestamp == datetime.strptime(Fechas[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
    
    # Interpolación entre el primer y último ciclo (a partir del 30 de Oct 23 uso PchipInterpolator)
    # interp_func = interp1d(t_full, T_full, kind='linear')
    # t_interp = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    # T_interp = np.round(interp_func(t_interp), 2)

    # interp_func_1 = CubicSpline(t_full, T_full)
    # t_interp_2 = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    # T_interp_2 = np.round(interp_func_2(t_interp_2), 2)

    interp_func = PchipInterpolator(t_full, T_full)
    t_interp = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    T_interp = np.round(interp_func(t_interp), 2)

    # Calcular t y T a partir de los datos
    t = np.round(delta_0 + np.cumsum(time_delta), 2)
    T = np.array([T_interp[np.flatnonzero(t_interp == t)[0]] for t in t])
    
    cmap = mpl.colormaps['jet'] #'viridis'
    if rango_T_fijo==True:
        norm_T = (np.array(T) - (-200)) / (50 - (-200))
    else:
        norm_T = (np.array(T) - np.array(T).min()) / (np.array(T).max() - np.array(T).min())
    
    colors = cmap(norm_T)
    
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    ax.plot(t_full,T_full,'.-',label=paths_T[0].split('/')[-1])
    ax.plot(t_interp,T_interp,'-',label='Temperatura interpolada')
    ax.scatter(t,T,color=colors,label='Temperatura muestra')

    plt.xlabel('t (s)')
    plt.ylabel('T (°C)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Temperatura de la muestra',fontsize=18)
    #plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=300,facecolor='w')
    plt.show()
    
    # Ajustar tiempos para que arranquen desde 0
    t = t - t_interp[0]
    t_interp = t_interp - t_interp[0]
    return t, T, t_interp, T_interp , colors,fig
    
#%% Implementacion
directorios_FF = [os.path.join(os.getcwd(),'C1',f) for f in os.listdir('C1') ]
directorios_FF.sort()
t_FF_1,T_FF_1,t_interp_FF_1,T_interp_FF_1,c_FF_1,_=procesar_temperatura(directorios_FF[0])
t_FF_2,T_FF_2,t_interp_FF_2,T_interp_FF_2,c_FF_2,_=procesar_temperatura(directorios_FF[1])
t_FF_3,T_FF_3,t_interp_FF_3,T_interp_FF_3,c_FF_3,_=procesar_temperatura(directorios_FF[2])
t_FF_4,T_FF_4,t_interp_FF_4,T_interp_FF_4,c_FF_4,_=procesar_temperatura(directorios_FF[3])

directorios_SV = [os.path.join(os.getcwd(),'SV',f) for f in os.listdir('SV') ]
directorios_SV.sort()
t_SV_1,T_SV_1,t_interp_SV_1,T_interp_SV_1,c_SV_1,_=procesar_temperatura(directorios_SV[0])
t_SV_2,T_SV_2,t_interp_SV_2,T_interp_SV_2,c_SV_2,_=procesar_temperatura(directorios_SV[1])
t_SV_3,T_SV_3,t_interp_SV_3,T_interp_SV_3,c_SV_3,_=procesar_temperatura(directorios_SV[2])

#%% Derivadas y ploteo 

dT_SV_1 = np.gradient(T_SV_1,t_SV_1)
dT_SV_2 = np.gradient(T_SV_2,t_SV_2)
dT_SV_3 = np.gradient(T_SV_3,t_SV_3)

dT_FF_1 = np.gradient(T_FF_1,t_FF_1)
dT_FF_2 = np.gradient(T_FF_2,t_FF_2)
dT_FF_3 = np.gradient(T_FF_3,t_FF_3)
dT_FF_4 = np.gradient(T_FF_4,t_FF_4)

indx_max_SV_1 = np.nonzero(dT_SV_1==max(dT_SV_1))
indx_max_SV_2 = np.nonzero(dT_SV_2==max(dT_SV_2))
indx_max_SV_3 = np.nonzero(dT_SV_3==max(dT_SV_3))
indx_max_FF_1 = np.nonzero(dT_FF_1==max(dT_FF_1))
indx_max_FF_2 = np.nonzero(dT_FF_2==max(dT_FF_2))
indx_max_FF_3 = np.nonzero(dT_FF_3==max(dT_FF_3))
indx_max_FF_4 = np.nonzero(dT_FF_4==max(dT_FF_4))

dT_interp_SV_1 = np.gradient(T_interp_SV_1,t_interp_SV_1)
dT_interp_SV_2 = np.gradient(T_interp_SV_2,t_interp_SV_2)
dT_interp_SV_3 = np.gradient(T_interp_SV_3,t_interp_SV_3)

dT_interp_FF_1 = np.gradient(T_interp_FF_1,t_interp_FF_1)
dT_interp_FF_2 = np.gradient(T_interp_FF_2,t_interp_FF_2)
dT_interp_FF_3 = np.gradient(T_interp_FF_3,t_interp_FF_3)
dT_interp_FF_4 = np.gradient(T_interp_FF_4,t_interp_FF_4)

#%% Grafico SV y FF+SV
fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(11,8),constrained_layout=True)

ax.plot(t_interp_SV_1,T_interp_SV_1,'-',label='T interp SV_1')
ax.plot(t_interp_SV_2,T_interp_SV_2,'-',label='T interp SV_2')
ax.plot(t_interp_SV_3,T_interp_SV_3,'-',label='T interp SV_3')

ax.scatter(t_SV_1,T_SV_1,color=c_SV_1,label='T SV_1',marker='.')
ax.scatter(t_SV_2,T_SV_2,color=c_SV_2,label='T SV_2',marker='.')
ax.scatter(t_SV_3,T_SV_3,color=c_SV_3,label='T SV_3',marker='.')

ax.scatter(t_SV_1[indx_max_SV_1],T_SV_1[indx_max_SV_1],marker='D',zorder=2,color='blue',label=f'dT/dt = {max(dT_SV_1):.2f} ºC/s')
ax.scatter(t_SV_2[indx_max_SV_2],T_SV_2[indx_max_SV_2],marker='D',zorder=2,color='orange',label=f'dT/dt = {max(dT_SV_2):.2f} ºC/s')
ax.scatter(t_SV_3[indx_max_SV_3],T_SV_3[indx_max_SV_3],marker='D',zorder=2,color='green',label=f'dT/dt = {max(dT_SV_3):.2f} ºC/s')

axin = ax.inset_axes([0.35, 0.15, 0.64, 0.58])
axin.plot(t_SV_1,dT_SV_1,'.-',lw=0.7,label='dT/dt SV_1')
axin.plot(t_SV_2,dT_SV_2,'.-',lw=0.7,label='dT/dt SV_2')
axin.plot(t_SV_3,dT_SV_3,'.-',lw=0.7,label='dT/dt SV_3')

ax2.plot(t_interp_FF_1,T_interp_FF_1,'-',label='T interp FF_1')
ax2.plot(t_interp_FF_3,T_interp_FF_3,'-',label='T interp FF_3')
ax2.plot(t_interp_FF_2,T_interp_FF_2,'-',label='T interp FF_2')
ax2.plot(t_interp_FF_4,T_interp_FF_4,'-',label='T interp FF_4')

ax2.scatter(t_FF_1,T_FF_1,color=c_FF_1,marker='.',label='T FF_1')
ax2.scatter(t_FF_2,T_FF_2,color=c_FF_2,marker='.',label='T FF_2')
ax2.scatter(t_FF_3,T_FF_3,color=c_FF_3,marker='.',label='T FF_3')
ax2.scatter(t_FF_4,T_FF_4,color=c_FF_4,marker='.',label='T FF_4')

ax2.scatter(t_FF_1[indx_max_FF_1],T_FF_1[indx_max_FF_1],marker='D',zorder=2,color='blue',label=f'dT/dt = {max(dT_FF_1):.2f} ºC/s')
ax2.scatter(t_FF_2[indx_max_FF_2],T_FF_2[indx_max_FF_2],marker='D',zorder=2,color='orange',label=f'dT/dt = {max(dT_FF_2):.2f} ºC/s')
ax2.scatter(t_FF_3[indx_max_FF_3],T_FF_3[indx_max_FF_3],marker='D',zorder=2,color='green',label=f'dT/dt = {max(dT_FF_3):.2f} ºC/s')

axin2 = ax2.inset_axes([0.35, 0.15, 0.64, 0.50])
axin2.plot(t_FF_1,dT_FF_1,'.-',lw=0.7,label='dT/dt FF_1')
axin2.plot(t_FF_2,dT_FF_2,'.-',lw=0.7,label='dT/dt FF_2')
axin2.plot(t_FF_3,dT_FF_3,'.-',lw=0.7,label='dT/dt FF_3')
axin2.plot(t_FF_4,dT_FF_4,'.-',lw=0.7,label='dT/dt FF_4')
for ai in [axin,axin2]:
    ai.set_xlabel('t (s)')
    ai.set_ylabel('dT/dt (ºC/s)')
    ai.legend(ncol=1)
    ai.grid()
    
for a in [ax,ax2]:
    a.grid()
    a.set_xlim(0,)
    a.set_ylabel('T (°C)')

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

ax.set_title('SV',loc='left')
ax2.set_title('SV + NP',loc='left')
ax2.set_xlabel('t (s)')
plt.show()


#%% Derivadas vs Temperatura
#%% trabajamos con SV3 y FF4 

fig,(ax)=plt.subplots(nrows=1,figsize=(10,4),constrained_layout=True,sharex=True)
# ax.plot(T_SV_1,dT_SV_1,'o-',label='dT/dt SV_1')
# ax.plot(T_SV_2,dT_SV_2,'o-',label='dT/dt SV_2')
ax.plot(T_SV_3,dT_SV_3,'o-',label='dT/dt SV_3')
# ax.plot(T_interp_SV_1,dT_interp_SV_1,'.-',label='dT/dt interp SV_1')
# ax.plot(T_interp_SV_2,dT_interp_SV_2,'.-',label='dT/dt interp SV_2')
# ax.plot(T_interp_SV_3,dT_interp_SV_3,'.-',label='dT/dt interp SV_3',zorder=0)
ax.plot(T_FF_1,dT_FF_1,'.-',label='dT/dt FF_1')
ax.plot(T_FF_2,dT_FF_2,'.-',label='dT/dt FF_2')
ax.plot(T_FF_3,dT_FF_3,'.-',label='dT/dt FF_3')
#ax.plot(T_FF_4,dT_FF_4,'o-',label='dT/dt FF_4')
# ax2.plot(T_interp_FF_4,dT_interp_FF_4,'o-',label='dT/dt interp_FF_4',zorder=0)
for ai in [ax]:
    ai.set_xlabel('T (°C)')
    ai.set_ylabel('dT/dt (ºC/s)')
    ai.legend(ncol=2)
    ai.grid()
plt.suptitle('Derivadas temporales vs Temperatura')
plt.show()

#%% Interpolaciones y restas - trabajamos con SV3 y FF4 

from scipy.interpolate import interp1d
f = interp1d(T_SV_3,dT_SV_3, fill_value='extrapolate')

_, indices_unicos = np.unique(T_SV_3, return_index=True)
indices_unicos_ordenados = np.sort(indices_unicos)
T_SV_3 =T_SV_3[indices_unicos_ordenados]
dT_SV_3 = dT_SV_3[indices_unicos_ordenados]

f2 = interp1d(T_SV_3,dT_SV_3, kind='cubic', fill_value='extrapolate')

xnew=T_FF_4[:-37]
ynew=f(xnew)
xnew2=T_FF_4[:-37]
ynew2=f2(xnew2)

resta_4 = dT_FF_4[:-37]-ynew2
T_FF_4_solido= T_FF_4[np.nonzero(T_FF_4<-5)]
# Calculo de SAR 
Concentracion = 14.92/1000 # m/m
CFF= 2.07 # J/Kg para la C1
CSV= 2.10 # J/Kg 

SAR_4 = (dT_FF_4[np.nonzero(T_FF_4<-5)]*CFF - ynew2[np.nonzero(T_FF_4<-5)]*CSV)/Concentracion
indx_max_resta_4= np.argwhere(resta_4==max(resta_4))[0][0]

from scipy.integrate import cumulative_trapezoid

# Recuperar T original
T_4_rec = cumulative_trapezoid(y=resta_4,x=t_FF_4[:-37],initial=0) + T_FF_4[0] 


fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)
ax.plot(T_SV_3,dT_SV_3,'.-',label='dT/dt SV_3')
ax.plot(T_FF_4,dT_FF_4,'o-',label='dT/dt FF_4')
# ax.plot(T_SV_3,dT_interp_4_new,'o-',label='dT/dt FF_4')
# ax.plot(xnew,ynew,'o-',label='dT/dt SV_3')
ax.plot(xnew2,ynew2,'o-',label='dT/dt FF_4 (interpolado)')
ax.plot(xnew2,resta_4,'o-',label='dTFF4/dt - dTSV3/dt')


ax2.plot(t_FF_4,T_FF_4,'.-',label='T FF4 ')
ax2.plot(t_SV_3[indices_unicos_ordenados],T_SV_3,'.-',label='T SV3')
ax2.plot(t_FF_4[:-37],T_4_rec,'.-',label='T FF4 s/atmosfera')
ax2.scatter(t_FF_4[indx_max_resta_4],T_4_rec[indx_max_resta_4],marker='D',color='tab:red',zorder=5,label=f'Max WR {resta_4[indx_max_resta_4]:.2f} °C/s')


ax2.set_xlim(0,160)
ax.set_xlabel('T (°C)')
ax2.set_xlabel('t (s)')
for ai in [ax,ax2]:
    ai.set_ylabel('T (ºC)')
    ai.legend(ncol=2)
    ai.grid()
ax.set_title('Derivadas temporales vs Temperatura',loc='left')
ax2.set_title('Temperatura vs t',loc='left')
plt.savefig('calentamiento_LN2_FF4_SV.png',dpi=300)
plt.show()

#%% FF3 - SV3 

f2 = interp1d(T_SV_3,dT_SV_3, kind='cubic', fill_value='extrapolate')

xnew=T_FF_3[:-45]
ynew=f(xnew)
xnew2=T_FF_3[:-45]
ynew2=f2(xnew2)

resta_3 = dT_FF_3[:-45]-ynew2
T_FF_3_solido= T_FF_3[np.nonzero(T_FF_3<-5)]
# Calculo de SAR 
Concentracion = 14.92/1000 # m/m
CFF= 2.07 # J/Kg para la C1
CSV= 2.10 # J/Kg 

SAR_3 = (dT_FF_3[np.nonzero(T_FF_3<-5)]*CFF - ynew2[np.nonzero(T_FF_3<-5)]*CSV)/Concentracion

indx_max_resta_3= np.argwhere(resta_3==max(resta_3))[0][0]

from scipy.integrate import cumulative_trapezoid

# Recuperar T original
T_FF3_rec = cumulative_trapezoid(y=resta_3,x=t_FF_3[:-45],initial=0) + T_FF_3[0] 
T_FF3_rec_max = T_FF3_rec[indx_max_resta_3]
T_FF_3_orig_max=T_FF_3[indx_max_resta_3]

fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)
ax.plot(T_SV_3,dT_SV_3,'.-',label='dT/dt SV_3')
ax.plot(T_FF_3,dT_FF_3,'o-',label='dT/dt FF_3')

# ax.plot(T_SV_3,dT_interp_3_new,'o-',label='dT/dt FF_3')
# ax.plot(xnew,ynew,'o-',label='dT/dt SV_3')
ax.plot(xnew2,ynew2,'o-',label='dT/dt FF_3 (interpolado)')
ax.plot(xnew2,resta_3,'o-',label='dTFF3/dt - dTSV3/dt')
ax.plot(xnew2[indx_max_resta_3],resta_3[indx_max_resta_3],'bD',label=f'Max WR {resta_3[indx_max_resta_3]:.2f} °C/s')


ax2.plot(t_FF_3,T_FF_3,'.-',label='T FF3 ')

ax2.plot(t_SV_3[indices_unicos_ordenados],T_SV_3,'.-',label='T SV3')
ax2.plot(t_FF_3[:-45],T_FF3_rec,'.-',label='T FF3 s/atmosfera')
ax2.scatter(t_FF_3[indx_max_resta_3],T_FF3_rec[indx_max_resta_3],marker='D',color='blue',zorder=5,label=f'Max WR {resta_3[indx_max_resta_3]:.2f} °C/s\n (at {T_FF3_rec[indx_max_resta_3]:.2f} °C)')


ax2.set_xlim(0,160)
ax.set_xlabel('T (°C)')
ax2.set_xlabel('t (s)')
for ai in [ax,ax2]:
    ai.set_ylabel('T (ºC)')
    ai.legend(ncol=2)
    ai.grid()
ax.set_title('Derivadas temporales vs Temperatura',loc='left')
ax2.set_title('Temperatura vs t',loc='left')
plt.savefig('calentamiento_LN2_FF3_SV.png',dpi=300)
plt.show()
#%% FF2 - SV3 

f2 = interp1d(T_SV_3,dT_SV_3, kind='cubic', fill_value='extrapolate')
xnew=T_FF_2[:-43]
ynew=f(xnew)
xnew2=T_FF_2[:-43]
ynew2=f2(xnew2)

resta_2 = dT_FF_2[:-43]-ynew2
T_FF_2_solido= T_FF_2[np.nonzero(T_FF_2<-5)]
# Calculo de SAR 
Concentracion = 14.92/1000 # m/m
CFF= 2.07 # J/Kg para la C1
CSV= 2.10 # J/Kg 

SAR_2 = (dT_FF_2[np.nonzero(T_FF_2<-5)]*CFF - ynew2[np.nonzero(T_FF_2<-5)]*CSV)/Concentracion
indx_max_resta_2= np.argwhere(resta_2==max(resta_2))[0][0]

from scipy.integrate import cumulative_trapezoid

# Recuperar T original
T_FF2_rec = cumulative_trapezoid(y=resta_2,x=t_FF_2[:-43],initial=0) + T_FF_2[0] 
T_FF2_rec_max = T_FF2_rec[indx_max_resta_2]
T_FF_2_orig_max=T_FF_2[indx_max_resta_2]

fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)
ax.plot(T_SV_3,dT_SV_3,'.-',label='dT/dt SV_3')
ax.plot(T_FF_2,dT_FF_2,'o-',label='dT/dt FF_2')

ax.plot(xnew2,ynew2,'o-',label='dT/dt FF_2 (interpolado)')
ax.plot(xnew2,resta_2,'o-',label='dTFF2/dt - dTSV3/dt')
ax.plot(xnew2[indx_max_resta_2],resta_2[indx_max_resta_2],'bD',label=f'Max WR {resta_2[indx_max_resta_2]:.2f} °C/s')


ax2.plot(t_FF_2,T_FF_2,'.-',label='T FF2 ')

ax2.plot(t_SV_3[indices_unicos_ordenados],T_SV_3,'.-',label='T SV3')
ax2.plot(t_FF_2[:-43],T_FF2_rec,'.-',label='T FF2 s/atmosfera')
ax2.scatter(t_FF_2[indx_max_resta_2],T_FF2_rec[indx_max_resta_2],marker='D',color='blue',zorder=5,label=f'Max WR {resta_2[indx_max_resta_2]:.2f} °C/s\n (at {T_FF2_rec[indx_max_resta_2]:.2f} °C)')


ax2.set_xlim(0,160)
ax.set_xlabel('T (°C)')
ax2.set_xlabel('t (s)')
for ai in [ax,ax2]:
    ai.set_ylabel('T (ºC)')
    ai.legend(ncol=2)
    ai.grid()
ax.set_title('Derivadas temporales vs Temperatura',loc='left')
ax2.set_title('Temperatura vs t',loc='left')
plt.savefig('calentamiento_LN2_FF2_SV.png',dpi=300)
plt.show()
#%% FF1 SV3
f2 = interp1d(T_SV_3,dT_SV_3, kind='cubic', fill_value='extrapolate')
xnew=T_FF_1[:-43]
ynew=f(xnew)
xnew2=T_FF_1[:-43]
ynew2=f2(xnew2)

resta_1 = dT_FF_1[:-43]-ynew2

indx_max_resta_1= np.argwhere(resta_1==max(resta_1))[0][0]
T_FF_1_solido= T_FF_1[np.nonzero(T_FF_1<-5)]
# Calculo de SAR 
Concentracion = 14.92/1000 # m/m
CFF= 2.07 # J/Kg para la C1
CSV= 2.10 # J/Kg 

SAR_1 = (dT_FF_1[np.nonzero(T_FF_1<-5)]*CFF - ynew2[np.nonzero(T_FF_1<-5)]*CSV)/Concentracion

from scipy.integrate import cumulative_trapezoid

# Recuperar T original
T_FF1_rec = cumulative_trapezoid(y=resta_1,x=t_FF_1[:-43],initial=0) + T_FF_1[0] 
T_FF1_rec_max = T_FF1_rec[indx_max_resta_1]
T_FF_1_orig_max=T_FF_1[indx_max_resta_1]

fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)
ax.plot(T_SV_3,dT_SV_3,'.-',label='dT/dt SV_3')
ax.plot(T_FF_1,dT_FF_1,'o-',label='dT/dt FF_1')

ax.plot(xnew2,ynew2,'o-',label='dT/dt FF_1 (interpolado)')
ax.plot(xnew2,resta_1,'o-',label='dTFF2/dt - dTSV3/dt')
ax.plot(xnew2[indx_max_resta_1],resta_1[indx_max_resta_1],'bD',label=f'Max WR {resta_1[indx_max_resta_1]:.2f} °C/s')


ax2.plot(t_FF_1,T_FF_1,'.-',label='T FF2 ')

ax2.plot(t_SV_3[indices_unicos_ordenados],T_SV_3,'.-',label='T SV3')
ax2.plot(t_FF_1[:-43],T_FF1_rec,'.-',label='T FF1 s/atmosfera')
ax2.scatter(t_FF_1[indx_max_resta_1],T_FF1_rec[indx_max_resta_1],marker='D',color='blue',zorder=5,label=f'Max WR {resta_1[indx_max_resta_1]:.2f} °C/s\n (at {T_FF2_rec[indx_max_resta_1]:.2f} °C)')

ax2.set_xlim(0,160)
ax.set_xlabel('T (°C)')
ax2.set_xlabel('t (s)')
for ai in [ax,ax2]:
    ai.set_ylabel('T (ºC)')
    ai.legend(ncol=2)
    ai.grid()
ax.set_title('Derivadas temporales vs Temperatura',loc='left')
ax2.set_title('Temperatura vs t',loc='left')
plt.savefig('calentamiento_LN2_FF1_SV.png',dpi=300)
plt.show()

#%% ahora todas las reconstrucciones

fig,(ax)=plt.subplots(nrows=1,figsize=(10,4),constrained_layout=True,sharex=True)
# ax.plot(t_SV_3,dT_SV_3,'o-',label='dT/dt SV_3')

ax.plot(t_FF_1[:-43],T_FF1_rec,'.-',label='T FF1 s/atmosfera')
ax.plot(t_FF_2[:-43],T_FF2_rec,'.-',label='T FF2 s/atmosfera')
ax.plot(t_FF_3[:-45],T_FF3_rec,'.-',label='T FF3 s/atmosfera')
ax.plot(t_FF_4[:-37],T_4_rec,'.-',label='T FF4 s/atmosfera')

#ax.plot(T_FF_4,dT_FF_4,'o-',label='dT/dt FF_4')
# ax2.plot(T_interp_FF_4,dT_interp_FF_4,'o-',label='dT/dt interp_FF_4',zorder=0)
for ai in [ax]:
    ai.set_ylabel('T (°C)')
    ai.set_xlabel('t (s)')
    ai.legend(ncol=2)
    ai.grid()
plt.suptitle('Cambio de Temperatura por NP - Enfriamiento poir LN2')
plt.savefig('cambio_T_por_NP_LN2.png',dpi=300)
plt.show()
#%%

fig,(ax)=plt.subplots(nrows=1,figsize=(10,4),constrained_layout=True,sharex=True)
# ax.plot(t_SV_3,dT_SV_3,'o-',label='dT/dt SV_3')

ax.plot(T_FF_1_solido,SAR_1,'.-',label='SAR FF1 (s/atmosfera)')
ax.plot(T_FF_2_solido,SAR_2,'.-',label='SAR FF2 (s/atmosfera)')
ax.plot(T_FF_3_solido,SAR_3,'.-',label='SAR FF3 (s/atmosfera)')
ax.plot(T_FF_4_solido,SAR_4,'.-',label='SAR FF4 (s/atmosfera)')

#ax.plot(T_FF_4,dT_FF_4,'o-',label='dT/dt FF_4')
# ax2.plot(T_interp_FF_4,dT_interp_FF_4,'o-',label='dT/dt interp_FF_4',zorder=0)
for ai in [ax]:
    ai.set_ylabel('SAR (W/g)')
    ai.set_xlabel('T (°C)')
    ai.legend(ncol=2)
    ai.grid()
plt.suptitle('SAR en funcion de la temperatura')
plt.savefig('SAR_vs_T.png',dpi=300)
plt.show()
# %%
