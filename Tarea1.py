# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:35:05 2021

@author: HP
"""
import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import math
data=pd.read_csv('CAUDAL_H@26057040.data',sep="|")
#print(data)
data['Fecha'] = pd.to_datetime(data['Fecha'])
#Cambiar indice por fecha
data = data.set_index('Fecha') 
#print(data)
#Agrupar por fechas y calcular promedios
AverageDailyData = data.groupby(pd.Grouper(freq='1D')).mean()
#Restablecer indices
AverageDailyData.reset_index(inplace=True)
#Exportar archivo con datos diarios promedio
AverageDailyData.to_csv('Promedios.data',index=False)
DailyDischarges=[]
for i in range(len(AverageDailyData)):
    DailyDischarges.append(AverageDailyData['Valor'][i])
#Limpiar lista de caudales
CleanDailyDischarges=[]
for i in range(0,len(DailyDischarges)): 
    if not math.isnan(DailyDischarges[i]):
        CleanDailyDischarges.append(DailyDischarges[i])
#print(CleanDailyDischarges)
#Verificando que no hay valores con nan
#for i in range(0,len(CleanDailyDischarges)):
    #if math.isnan(CleanDailyDischarges[i]):
        #print('Error. Hay valores con nan')
    #elif i==len(CleanDailyDischarges)-1:
        #print('Todo salió bien')
#print('El número de datos limpios es:', len(CleanDailyDischarges), 'mientras el número de datos totales es:', len (DailyDischarges))
#Creando lista con frontera de intervalos
IntervalBorders=[]
for i in range(0,30):
    IntervalBorders.append([4*i,4*(i+1)])
#Creando contadores
IntervalCounters=[]
for i in range(0,30):
    l=[]
    IntervalCounters.append(l)    
#Creando listas de valores dentro de los intervalos
for i in range(0,len(CleanDailyDischarges)):
    for j in range(0,len(IntervalBorders)):    
        if IntervalBorders[j][0] <= CleanDailyDischarges[i] and CleanDailyDischarges[i] < IntervalBorders[j][1]:
            IntervalCounters[j].append(CleanDailyDischarges[i])
#Creando la lista de frecuencias
Frequencies=[]            
for i in range(0, len(IntervalCounters)):
    if len(IntervalCounters)!=0: 
        Frequencies.append(len(IntervalCounters[i]))
#print(Frequencies)
#Creando lista de frecuencias acumuladas
AcumFrequencies=[]
s=0
for i in range(0,len(Frequencies)):
    s=s+Frequencies[i]
    AcumFrequencies.append(s)
#print(AcumFrequencies)
Intervals=[]
for i in range(1,31):
    Intervals.append(i)
#print(IntervalBorders)
RelativeFrequencies=[]
for i in range(0,len(Frequencies)):
    RelativeFrequencies.append(Frequencies[i]/sum(Frequencies)*100)
AcumProbabilities=[]
for i in range(0,len(AcumFrequencies)):
    AcumProbabilities.append(AcumFrequencies[i])
plt.close('all')
#Creando histograma de frecuencias absolutas
plt.figure('Fig.1')
plt.plot(Intervals,Frequencies)
plt.bar(Intervals,Frequencies,color='g')
plt.grid(True)
plt.xticks(range(1,31))
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.close('Fig.1')

plt.figure('Fig.2')
plt.plot(Intervals,AcumFrequencies)
plt.grid(True)
plt.xticks(range(1,31))
plt.xlabel('Clases')
plt.ylabel('Frecuencia acumulada')
plt.close('Fig.2')


#print('El máximo caudal diario promedio es ', max(DailyDischarges), 'm3/s')
#print('El mínimo caudal diario promedio es ', min(DailyDischarges), 'm3/s')
Media=stat.mean(CleanDailyDischarges)
StandardDeviation=AverageDailyData.std()['Valor']

#Coeficiente de exceso de curtosis.
Kurtosis=AverageDailyData.kurt()['Valor']

#Coeficiente de asimetría
Skewness=AverageDailyData.skew()['Valor']

#Función de distribución de probabilidad acumulada

AverageDailyData.hist(cumulative=True,density=1,bins=10000)
plt.xticks(range(0,101,10))
plt.title('Función de distribución de probabilidad acumulada')
plt.xlabel('Caudal ($m^3/s$)')
print(AverageDailyData)

#Función de densidad de probabilidad
#AverageDailyData.hist(cumulative=False,normed=True,bins=1000)

#Creando lista de promedios mensuales
AverageDailyData2=AverageDailyData.set_index('Fecha')
AverageMonthlyData=AverageDailyData2.groupby(pd.Grouper(freq='1M')).mean()
AverageMonthlyData.reset_index(inplace=True)
AverageMonthlyData.to_csv('PromediosMensuales.data',index=False)

#Medias y desviaciones estándar anuales
AverageMonthlyData2=AverageMonthlyData.set_index('Fecha')
AverageMonthlyData2.reset_index(inplace=True)
AverageMonthlyData2['Fecha']=pd.to_datetime(AverageMonthlyData2['Fecha'])
AverageMonthlyData2['Fecha']=AverageMonthlyData2['Fecha'].apply(lambda x: x.strftime('%Y-%m'))
#print('La media es:',Media)
#print('La desviación estándar es',StandardDeviation)
#print('El coeficiente de kurtosis es:',Kurtosis)
#print('El coeficiente de asimetría es:',Skewness)
#print(AnnualStandardDeviations)

#Función de autocorrelación diaria
DailyACFy=[]
for i in range(0,181):
    DailyACFyy=AverageDailyData['Valor'].autocorr(lag=i)
    DailyACFy.append(DailyACFyy)

##Gráfica Autocorrelación diaria    
#plt.figure('Fig.8')
#plt.plot(range(0,181),DailyACFy,'ro')
#plt.grid(True)
#plt.xticks(range(0,181,10))
#plt.xlabel('Rezago (Días)')
#plt.ylabel('Coeficiente de correlación')

#Función de autocorrelación mensual
MonthlyACFy=[]
for i in range(0,16):
    MonthlyACFyy=AverageMonthlyData['Valor'].autocorr(lag=i)
    MonthlyACFy.append(MonthlyACFyy)

##Grafica Autocorrelación mensual    
#plt.figure('Fig.9')
#plt.plot(range(0,16),MonthlyACFy,marker='o',color='b')
#plt.grid(True)
#plt.xticks(range(0,16))
#plt.xlabel('Rezago (Meses)')
#plt.ylabel('Coeficiente de correlación')    

#AverageMonthlyData2.plot()
#print(AverageMonthlyData2)

#Datos de ciclo anual
AnualCycleMatrix=pd.read_csv('PromediosMensuales2.csv',sep=",")
#MonthMeans=pd.DataFrame({"Ano":["Promedio"],"1":[AnualCycleMatrix['1'].mean()],"2":[AnualCycleMatrix['2'].mean()],"3":[AnualCycleMatrix['3'].mean()],"4":[AnualCycleMatrix['4'].mean()],"5":[AnualCycleMatrix['5'].mean()],"6":[AnualCycleMatrix['6'].mean()],"7":[AnualCycleMatrix['7'].mean()],"8":[AnualCycleMatrix['8'].mean()],"9":[AnualCycleMatrix['9'].mean()],"10":[AnualCycleMatrix['10'].mean()],"11":[AnualCycleMatrix['11'].mean()],"12":[AnualCycleMatrix['12'].mean()],})
#MonthMeans[sorted(MonthMeans.columns)]
AnualCycleMatrix2=AnualCycleMatrix
del AnualCycleMatrix2['Ano'] 

MediasMensuales=[]
MediasMensuales.append(AnualCycleMatrix2["1"][45])
MediasMensuales.append(AnualCycleMatrix2["2"][45])
MediasMensuales.append(AnualCycleMatrix2["3"][45])
MediasMensuales.append(AnualCycleMatrix2["4"][45])
MediasMensuales.append(AnualCycleMatrix2["5"][45])
MediasMensuales.append(AnualCycleMatrix2["6"][45])
MediasMensuales.append(AnualCycleMatrix2["7"][45])
MediasMensuales.append(AnualCycleMatrix2["8"][45])
MediasMensuales.append(AnualCycleMatrix2["9"][45])
MediasMensuales.append(AnualCycleMatrix2["10"][45])
MediasMensuales.append(AnualCycleMatrix2["11"][45])
MediasMensuales.append(AnualCycleMatrix2["12"][45])

DesvMensuales=[]
DesvMensuales.append(AnualCycleMatrix2["1"][46])
DesvMensuales.append(AnualCycleMatrix2["2"][46])
DesvMensuales.append(AnualCycleMatrix2["3"][46])
DesvMensuales.append(AnualCycleMatrix2["4"][46])
DesvMensuales.append(AnualCycleMatrix2["5"][46])
DesvMensuales.append(AnualCycleMatrix2["6"][46])
DesvMensuales.append(AnualCycleMatrix2["7"][46])
DesvMensuales.append(AnualCycleMatrix2["8"][46])
DesvMensuales.append(AnualCycleMatrix2["9"][46])
DesvMensuales.append(AnualCycleMatrix2["10"][46])
DesvMensuales.append(AnualCycleMatrix2["11"][46])
DesvMensuales.append(AnualCycleMatrix2["12"][46])

ErrMensuales=[]
ErrMensuales.append(AnualCycleMatrix2["1"][47])
ErrMensuales.append(AnualCycleMatrix2["2"][47])
ErrMensuales.append(AnualCycleMatrix2["3"][47])
ErrMensuales.append(AnualCycleMatrix2["4"][47])
ErrMensuales.append(AnualCycleMatrix2["5"][47])
ErrMensuales.append(AnualCycleMatrix2["6"][47])
ErrMensuales.append(AnualCycleMatrix2["7"][47])
ErrMensuales.append(AnualCycleMatrix2["8"][47])
ErrMensuales.append(AnualCycleMatrix2["9"][47])
ErrMensuales.append(AnualCycleMatrix2["10"][47])
ErrMensuales.append(AnualCycleMatrix2["11"][47])
ErrMensuales.append(AnualCycleMatrix2["12"][47])

##Grafica de ciclo anual
#plt.figure('Ciclo anual')
#plt.errorbar(range(1,13),MediasMensuales,yerr=ErrMensuales,color='k',ecolor='r',capsize=4)
#plt.xticks(range(1,13))
#plt.grid(True)
#plt.xlabel('Mes')
#plt.ylabel('Caudal mensual promedio ($m^3/s$)')  


#Datos ciclo anual estandarizado
StandardAnnualCycle=pd.read_csv('CaudalesEst.csv',sep=",")
StandardAnnualCycle['Mes']=pd.to_datetime(StandardAnnualCycle['Mes'])
StandardAnnualCycle['Mes']=StandardAnnualCycle['Mes'].apply(lambda x: x.strftime('%Y-%m'))
StandardAnnualCycleticks=[0,48,168,288,408,528]

AverageList=[]
for i in range(0,len(StandardAnnualCycle)):
    AverageList.append(StandardAnnualCycle['CaudalEst'][i])

AverageDates=[]
for i in range(0,len(AverageMonthlyData)):
    AverageDates.append(AverageMonthlyData['Fecha'][i])
    
StandardAnnualDates=[]
for i in range(0,len(AverageMonthlyData)):
    StandardAnnualDates.append(StandardAnnualCycle['Mes'][i])    

#Fechas de string a fecha (pinche formato de fechas de pandas)
#converted_dates = list(map(datetime.datetime.strptime, StandardAnnualDates, len(StandardAnnualDates)*['%Y-%m']))
#x_axis = converted_dates
#formatter = dates.DateFormatter('%Y-%m')

###Grafica ciclo multianual estandarizado
#plt.figure('Ciclo anual estandarizado')
#plt.plot(range(len(AverageList)),AverageList,label='Ciclo anual estandarizado')
#plt.xticks(StandardAnnualCycleticks,['1976-01','1980-01','1990-01','2000-01','2010-01','2020-01'])
#plt.xlim(0,532)
##ax = plt.gcf().axes[0] 
##ax.xaxis.set_major_formatter(formatter)
#plt.grid(True)
#plt.xlabel('Fecha')
#plt.ylabel('Caudal estandarizado') 
    
#Medias Estandarizadas
MediasEst=pd.read_csv('MediasEst.csv',sep=",")

#Desviaciones Estandarizadas
DesviacionesEst=pd.read_csv('DesviacionesEst.csv',sep=",")    

#Grafica ciclo anual estandarizado con desviaciones estandar

#plt.figure('Ciclo anual de caudales mensuales estandarizados')
#plt.plot(range(1,13),[0,0,0,0,0,0,0,0,0,0,0,0],marker='o',color='r',label='Caudal mensual promedio estandarizado')
#plt.plot(range(1,13),[1,1,1,1,1,1,1,1,1,1,1,1],marker='o',color='b',label='Desviación estándar mensual')
#plt.xticks(range(1,13))
#plt.grid(True)
#plt.xlabel('Mes')
#plt.legend(loc='center right')    

SOIData=pd.read_csv('soidef.csv',sep=",")

SOIindex=[]
for i in range(len(SOIData)):
    SOIindex.append(SOIData['SOI'][i])

ONIData=pd.read_csv('ONI.csv',sep=",")

ONIindex=[]
for i in range(len(ONIData)):
    ONIindex.append(ONIData['ONI'][i])
    
MEIData=pd.read_csv('MEI.csv',sep=",")    
 
MEIindex=[]
for i in range(len(MEIData)):
    MEIindex.append(MEIData['MEI'][i])
    
##Grafica ciclo anual estandarizado con SOI, ONI, MEI    
#plt.figure('Ciclo anual estandarizado con indices del ENSO')
#plt.plot(range(len(AverageList)),AverageList,label='Caudal medio mensual estandarizado')
#plt.plot(range(len(AverageList)),SOIindex,label='SOI')
#plt.plot(range(len(AverageList)),ONIindex,label='ONI')
#plt.plot(range(len(AverageList)),MEIindex,label='MEI')
#plt.xticks(StandardAnnualCycleticks,['1976-01','1980-01','1990-01','2000-01','2010-01','2020-01'])
#plt.xlim(0,532)
#plt.grid(True)
#plt.xlabel('Fecha')
#plt.legend(loc='upper right')

NAOData=pd.read_csv('NAO.csv',sep=",")

NAOindex=[]
for i in range(len(NAOData)):
    NAOindex.append(NAOData['NAO'][i])


##Grafica ciclo anual estandarizado con NAO
#plt.figure('Ciclo anual estandarizado comparado con NAO')
#plt.plot(range(len(AverageList)),AverageList,label='Caudal medio mensual estandarizado')
#plt.plot(range(len(AverageList)),NAOindex,label='NAO')
#plt.xticks(StandardAnnualCycleticks,['1976-01','1980-01','1990-01','2000-01','2010-01','2020-01'])
#plt.xlim(0,532)
##ax = plt.gcf().axes[0] 
##ax.xaxis.set_major_formatter(formatter)
#plt.grid(True)
#plt.xlabel('Fecha')
#plt.legend(loc='upper right')

    
PDOData=pd.read_csv('PDO.csv',sep=",")

PDOindex=[]
for i in range(len(PDOData)):
    PDOindex.append(PDOData['PDO'][i])    
    
#Grafica ciclo anual estandarizado con PDO
plt.figure('Ciclo anual estandarizado comparado con PDO')
plt.plot(range(len(AverageList)),AverageList,label='Caudal medio mensual estandarizado')
plt.plot(range(len(AverageList)),PDOindex,label='PDO')
plt.xticks(StandardAnnualCycleticks,['1976-01','1980-01','1990-01','2000-01','2010-01','2020-01'])
plt.xlim(0,532)
#ax = plt.gcf().axes[0] 
#ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel('Fecha')
plt.legend(loc='upper right')

SOIIndex2=[]
ONIIndex2=[]
StandardListNoMEI=[]
for i in range(0,len(AverageList)): 
    if not math.isnan(AverageList[i]):
        StandardListNoMEI.append(AverageList[i])
        SOIIndex2.append(SOIindex[i])
        ONIIndex2.append(ONIindex[i])

CleanMEI1=[]
StandardListMEI1=[]
for i in range(36,len(MEIindex)):
    CleanMEI1.append(MEIindex[i])
    StandardListMEI1.append(AverageList[i])

CleanMEI=[]
StandardListMEI=[]
for i in range(0,len(StandardListMEI1)): 
    if not math.isnan(StandardListMEI1[i]):
        StandardListMEI.append(StandardListMEI1[i])
        CleanMEI.append(CleanMEI1[i])
        

def autocorrS (rezago, Caudales_m, Caudales_n):
    tau_m = []
    for t in range(rezago+1):
        r = 0
        p1 = 0
        p2 = 0
        p3 = 0
        for i in range(len(Caudales_m)):
            if i + t >= len(Caudales_m):
                break
            mi = sum(Caudales_m[i:])/len(Caudales_m[i:])
            mii = sum(Caudales_n[i+t:])/len(Caudales_n[i+t:])
            p11 = (Caudales_m[i] - mi)*(Caudales_n[i+t] - mii)
            p1 = p1 + p11
            p22 = (Caudales_m[i] - mi)**2
            p2 = p2 + p22
            p33 = (Caudales_n[i+t] - mii)**2
            p3 = p3 + p33
        r = p1/((p2*p3)**(1/2))
        tau_m.append(r)
    return tau_m

rezago = 5
autocoSOI = autocorrS(rezago,SOIIndex2,StandardListNoMEI)
autocoMEI = autocorrS(rezago,CleanMEI,StandardListMEI)
autocoONI = autocorrS(rezago,ONIIndex2,StandardListNoMEI)

## Autocorrelacion SOI
#plt.figure('Fig_14')
#plt.plot(range(rezago+1),autocoSOI,color='k',marker='o')
#plt.grid()
#plt.xlabel('Rezago [meses]')
#plt.ylabel('Coef. de Auto-correlación')

## Autocorrelacion ONI
#plt.figure('Fig_14.3')
#plt.plot(range(rezago+1),autocoONI,color='k',marker='o')
#plt.grid()
#plt.xlabel('Rezago [meses]')
#plt.ylabel('Coef. de Auto-correlación')

## Autocorrelacion MEIv2
#plt.figure('Fig_14.2')
#plt.plot(range(rezago+1),autocoMEI,color='k',marker='o')
#plt.grid()
#plt.xlabel('Rezago [meses]')
#plt.ylabel('Coef. de Auto-correlación')

# Las 3 auto-correlaciones juntas
#plt.figure('Fig_14.4')
#plt.plot(range(rezago+1),autocoSOI,color='limegreen',marker='o',label='Corr. SOI')
#plt.plot(range(rezago+1),autocoMEI,color='mediumblue',marker='o',label='Corr. MEIv2')
#plt.plot(range(rezago+1),autocoONI,color='tab:orange',marker='o',label='Corr. ONI')
#plt.grid()
#plt.legend()
#plt.xlabel('Rezago [meses]')
#plt.ylabel('Coef. de Auto-correlación')

CaudalesNinoData=pd.read_csv('CaudalesNino.csv',sep=",")
ErroresCaudalesNinoData=pd.read_csv('ErroresCaudalesNino.csv',sep=",")

CaudalesNino=[]
for i in range(0,len(CaudalesNinoData)):
    CaudalesNino.append(CaudalesNinoData['Nino'][i])
    
CaudalesNina=[]
for i in range(0,len(CaudalesNinoData)):
    CaudalesNina.append(CaudalesNinoData['Nina'][i])    

CaudalesNeutro=[]
for i in range(0,len(CaudalesNinoData)):
    CaudalesNeutro.append(CaudalesNinoData['Neutro'][i])
    
ErroresCaudalesNino=[]
for i in range(0,len(ErroresCaudalesNinoData)):
    ErroresCaudalesNino.append(ErroresCaudalesNinoData['Nino'][i])
    
ErroresCaudalesNina=[]
for i in range(0,len(ErroresCaudalesNinoData)):
    ErroresCaudalesNina.append(ErroresCaudalesNinoData['Nina'][i])    

ErroresCaudalesNeutro=[]
for i in range(0,len(CaudalesNinoData)):
    ErroresCaudalesNeutro.append(ErroresCaudalesNinoData['Neutro'][i])    

##Grafica de ciclo anual en Niño
#plt.figure('Ciclo anual en eventos Nino')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNino,yerr=ErroresCaudalesNino,color='k',ecolor='r',capsize=4,label='Nino')
#plt.xticks(range(len(CaudalesNino)),['Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar','Abr','May'])
#plt.grid(True)
#plt.xlabel('Mes')
#plt.ylabel('Caudal mensual promedio ($m^3/s$)') 
#plt.legend(loc='upper left')

##Grafica de ciclo anual en Niña
#plt.figure('Ciclo anual en eventos Nina')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNina,yerr=ErroresCaudalesNina,color='k',ecolor='r',capsize=4,label='Nina')
#plt.xticks(range(len(CaudalesNino)),['Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar','Abr','May'])
#plt.grid(True)
#plt.xlabel('Mes')
#plt.ylabel('Caudal mensual promedio ($m^3/s$)') 
#plt.legend(loc='upper left')

##Grafica de ciclo anual en Neutro
#plt.figure('Ciclo anual en eventos Neutro')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNeutro,yerr=ErroresCaudalesNeutro,color='k',ecolor='r',capsize=4,label='Neutro')
#plt.xticks(range(len(CaudalesNino)),['Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar','Abr','May'])
#plt.grid(True)
#plt.xlabel('Mes')
#plt.ylabel('Caudal mensual promedio ($m^3/s$)') 
#plt.legend(loc='upper left')  
    
##Grafica de ciclo anual en Niño, Niña y Neutro
#plt.figure('Ciclo anual en eventos Niño, Niña y Neutral')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNino,yerr=ErroresCaudalesNino,color='r',ecolor='r',capsize=4,label='Niño')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNina,yerr=ErroresCaudalesNina,color='b',ecolor='b',capsize=4,label='Niña')
#plt.errorbar(range(len(CaudalesNino)),CaudalesNeutro,yerr=ErroresCaudalesNeutro,color='g',ecolor='g',capsize=4,label='Neutral')    
#plt.xticks(range(len(CaudalesNino)),['Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar','Abr','May'])
#plt.grid(True)
#plt.xlabel('Mes')
#plt.ylabel('Caudal mensual promedio ($m^3/s$)') 
#plt.legend(loc='upper left')    

ONICorrData=pd.read_csv('ONIcorr.csv',sep=",")
MEICorrData=pd.read_csv('MEIcorr.csv',sep=",")
SOICorrData=pd.read_csv('SOIcorr.csv',sep=",")
NAOCorrData=pd.read_csv('NAOcorr.csv',sep=",")    
PDOCorrData=pd.read_csv('PDOcorr.csv',sep=",")

#Lista de datos correlaciones

#ONI
ONI_DEF=[]
ONI_MAM=[]
ONI_JJA=[]
ONI_SON=[]
for i in range(len(ONICorrData)):
    ONI_DEF.append(ONICorrData['DEF'][i])
    ONI_MAM.append(ONICorrData['MAM'][i])
    ONI_JJA.append(ONICorrData['JJA'][i])
    ONI_SON.append(ONICorrData['SON'][i])
    
#MEI
MEI_DEF=[]
MEI_MAM=[]
MEI_JJA=[]
MEI_SON=[]
for i in range(len(MEICorrData)):
    MEI_DEF.append(MEICorrData['DEF'][i])
    MEI_MAM.append(MEICorrData['MAM'][i])
    MEI_JJA.append(MEICorrData['JJA'][i])
    MEI_SON.append(MEICorrData['SON'][i])
    
#SOI
SOI_DEF=[]
SOI_MAM=[]
SOI_JJA=[]
SOI_SON=[]
for i in range(len(SOICorrData)):
    SOI_DEF.append(SOICorrData['DEF'][i])
    SOI_MAM.append(SOICorrData['MAM'][i])
    SOI_JJA.append(SOICorrData['JJA'][i])
    SOI_SON.append(SOICorrData['SON'][i])

#NAO
NAO_DEF=[]
NAO_MAM=[]
NAO_JJA=[]
NAO_SON=[]
for i in range(len(NAOCorrData)):
    NAO_DEF.append(NAOCorrData['DEF'][i])
    NAO_MAM.append(NAOCorrData['MAM'][i])
    NAO_JJA.append(NAOCorrData['JJA'][i])
    NAO_SON.append(NAOCorrData['SON'][i])    
    
#PDO
PDO_DEF=[]
PDO_MAM=[]
PDO_JJA=[]
PDO_SON=[]
for i in range(len(PDOCorrData)):
    PDO_DEF.append(PDOCorrData['DEF'][i])
    PDO_MAM.append(PDOCorrData['MAM'][i])
    PDO_JJA.append(PDOCorrData['JJA'][i])
    PDO_SON.append(PDOCorrData['SON'][i])

#Graficas de correlacion

##ONI
#marcas_x=['DEF','MAM','JJA','SON']
#xrange_DEF=[-0.3,0.7,1.7,2.7]
#xrange_MAM=[-0.1,0.9,1.9,2.9]
#xrange_JJA=[0.1,1.1,2.1,3.1]
#xrange_SON=[0.3,1.3,2.3,3.3]
#plt.figure('Correlacion ONI')
#plt.bar(xrange_DEF,ONI_DEF,width=0.2,color='b',align='center',label='DEF')    
#plt.bar(xrange_MAM,ONI_MAM,width=0.2,color='r',align='center',label='MAM')    
#plt.bar(xrange_JJA,ONI_JJA,width=0.2,color='g',align='center',label='JJA')
#plt.bar(xrange_SON,ONI_SON,width=0.2,color='m',align='center',label='SON')
#plt.grid(axis='x')
#plt.xticks([0,1,2,3],marcas_x)
#plt.xlabel('Trimestre')
#plt.ylabel('Coef. de correlación')
#plt.legend(loc='upper right')   

##MEI
#marcas_x=['DEF','MAM','JJA','SON']
#xrange_DEF=[-0.3,0.7,1.7,2.7]
#xrange_MAM=[-0.1,0.9,1.9,2.9]
#xrange_JJA=[0.1,1.1,2.1,3.1]
#xrange_SON=[0.3,1.3,2.3,3.3]
#plt.figure('Correlacion MEI')
#plt.bar(xrange_DEF,MEI_DEF,width=0.2,color='b',align='center',label='DEF')    
#plt.bar(xrange_MAM,MEI_MAM,width=0.2,color='r',align='center',label='MAM')    
#plt.bar(xrange_JJA,MEI_JJA,width=0.2,color='g',align='center',label='JJA')
#plt.bar(xrange_SON,MEI_SON,width=0.2,color='m',align='center',label='SON')
#plt.grid(axis='x')
#plt.xticks([0,1,2,3],marcas_x)
#plt.xlabel('Trimestre')
#plt.ylabel('Coef. de correlación')
#plt.legend(loc='upper right')

##SOI
#marcas_x=['DEF','MAM','JJA','SON']
#xrange_DEF=[-0.3,0.7,1.7,2.7]
#xrange_MAM=[-0.1,0.9,1.9,2.9]
#xrange_JJA=[0.1,1.1,2.1,3.1]
#xrange_SON=[0.3,1.3,2.3,3.3]
#plt.figure('Correlacion SOI')
#plt.bar(xrange_DEF,SOI_DEF,width=0.2,color='b',align='center',label='DEF')    
#plt.bar(xrange_MAM,SOI_MAM,width=0.2,color='r',align='center',label='MAM')    
#plt.bar(xrange_JJA,SOI_JJA,width=0.2,color='g',align='center',label='JJA')
#plt.bar(xrange_SON,SOI_SON,width=0.2,color='m',align='center',label='SON')
#plt.grid(axis='x')
#plt.xticks([0,1,2,3],marcas_x)
#plt.xlabel('Trimestre')
#plt.ylabel('Coef. de correlación')
#plt.legend(loc='upper right')

##NAO
#marcas_x=['DEF','MAM','JJA','SON']
#xrange_DEF=[-0.3,0.7,1.7,2.7]
#xrange_MAM=[-0.1,0.9,1.9,2.9]
#xrange_JJA=[0.1,1.1,2.1,3.1]
#xrange_SON=[0.3,1.3,2.3,3.3]
#plt.figure('Correlacion NAO')
#plt.bar(xrange_DEF,NAO_DEF,width=0.2,color='b',align='center',label='DEF')    
#plt.bar(xrange_MAM,NAO_MAM,width=0.2,color='r',align='center',label='MAM')    
#plt.bar(xrange_JJA,NAO_JJA,width=0.2,color='g',align='center',label='JJA')
#plt.bar(xrange_SON,NAO_SON,width=0.2,color='m',align='center',label='SON')
#plt.grid(axis='x')
#plt.xticks([0,1,2,3],marcas_x)
#plt.xlabel('Trimestre')
#plt.ylabel('Coef. de correlación')
#plt.legend(loc='upper right')

#PDO
marcas_x=['DEF','MAM','JJA','SON']
xrange_DEF=[-0.3,0.7,1.7,2.7]
xrange_MAM=[-0.1,0.9,1.9,2.9]
xrange_JJA=[0.1,1.1,2.1,3.1]
xrange_SON=[0.3,1.3,2.3,3.3]
plt.figure('Correlacion PDO')
plt.bar(xrange_DEF,PDO_DEF,width=0.2,color='b',align='center',label='DEF')    
plt.bar(xrange_MAM,PDO_MAM,width=0.2,color='r',align='center',label='MAM')    
plt.bar(xrange_JJA,PDO_JJA,width=0.2,color='g',align='center',label='JJA')
plt.bar(xrange_SON,PDO_SON,width=0.2,color='m',align='center',label='SON')
plt.grid(axis='x')
plt.xticks([0,1,2,3],marcas_x)
plt.xlabel('Trimestre')
plt.ylabel('Coef. de correlación')
plt.legend(loc='upper right')       

#Datos para Histogramas Nino, Nina, Neutro
HistoNinoData=pd.read_csv('HistoNinoData.csv',sep=",")
HistoNinaData=pd.read_csv('HistoNinaData.csv',sep=",")
HistoNeutroData=pd.read_csv('HistoNeutroData.csv',sep=",")

HistoIntervalos=[]
for i in range(0,12):
    HistoIntervalos.append([5*i,5*(i+1)])

#Creando contadores
HistoCounters_Nino=[]
HistoCounters_Nina=[]
HistoCounters_Neutro=[]
for i in range(0,12):
    l=[]
    HistoCounters_Nino.append(l)
    
for i in range(0,12):
    l=[]    
    HistoCounters_Nina.append(l)
    
for i in range(0,12):
    l=[]    
    HistoCounters_Neutro.append(l)
    
#Creando listas de valores dentro de los intervalos para Nino
for i in range(0,len(HistoNinoData)):
    for j in range(0,len(HistoIntervalos)):    
        if HistoIntervalos[j][0] <= HistoNinoData['Nino'][i] and HistoNinoData['Nino'][i] < HistoIntervalos[j][1]:
            HistoCounters_Nino[j].append(HistoNinoData['Nino'][i])
            
#Creando listas de valores dentro de los intervalos para Nina
for i in range(0,len(HistoNinaData)):
    for j in range(0,len(HistoIntervalos)):    
        if HistoIntervalos[j][0] <= HistoNinaData['Nina'][i] and HistoNinaData['Nina'][i] < HistoIntervalos[j][1]:
            HistoCounters_Nina[j].append(HistoNinaData['Nina'][i])

#Creando listas de valores dentro de los intervalos para Neutro
for i in range(0,len(HistoNeutroData)):
    for j in range(0,len(HistoIntervalos)):    
        if HistoIntervalos[j][0] <= HistoNeutroData['Neutro'][i] and HistoNeutroData['Neutro'][i] < HistoIntervalos[j][1]:
            HistoCounters_Neutro[j].append(HistoNeutroData['Neutro'][i])            

#Creando la lista de frecuencias para Nino
Frequencies_Nino=[]            
for i in range(0, len(HistoCounters_Nino)):
    if len(HistoCounters_Nino)!=0: 
        Frequencies_Nino.append(len(HistoCounters_Nino[i]))
        
#Creando la lista de frecuencias para Nina
Frequencies_Nina=[]            
for i in range(0, len(HistoCounters_Nina)):
    if len(HistoCounters_Nina)!=0: 
        Frequencies_Nina.append(len(HistoCounters_Nina[i]))       
        
#Creando la lista de frecuencias para Neutro
Frequencies_Neutro=[]            
for i in range(0, len(HistoCounters_Neutro)):
    if len(HistoCounters_Neutro)!=0: 
        Frequencies_Neutro.append(len(HistoCounters_Neutro[i]))        

##Creando histograma de frecuencias absolutas para Nino
#plt.figure('Histograma Nino')
#plt.plot(range(1,13),Frequencies_Nino)
#plt.bar(range(1,13),Frequencies_Nino,color='g')
#plt.grid(True)
#plt.xticks(range(1,13))
#plt.xlabel('Clases')
#plt.ylabel('Frecuencia')

##Creando histograma de frecuencias absolutas para Nino
#plt.figure('Histograma Nina')
#plt.plot(range(1,13),Frequencies_Nina)
#plt.bar(range(1,13),Frequencies_Nina,color='g')
#plt.grid(True)
#plt.xticks(range(1,13))
#plt.xlabel('Clases')
#plt.ylabel('Frecuencia')

##Creando histograma de frecuencias absolutas para Neutro
#plt.figure('Histograma Neutro')
#plt.plot(range(1,13),Frequencies_Neutro)
#plt.bar(range(1,13),Frequencies_Neutro,color='g')
#plt.grid(True)
#plt.xticks(range(1,13))
#plt.xlabel('Clases')
#plt.ylabel('Frecuencia')

#Medias
Media_Nino=HistoNinoData.mean()['Nino']
Media_Nina=HistoNinaData.mean()['Nina']
Media_Neutro=HistoNeutroData.mean()['Neutro']

#Varianzas
Varianza_Nino=HistoNinoData.var()['Nino']
Varianza_Nina=HistoNinaData.var()['Nina']
Varianza_Neutro=HistoNeutroData.var()['Neutro']

#Asimetrias
Asimetria_Nino=HistoNinoData.skew()['Nino']
Asimetria_Nina=HistoNinaData.skew()['Nina']
Asimetria_Neutro=HistoNeutroData.skew()['Neutro']

#Curtosis
Curtosis_Nino=HistoNinoData.kurt()['Nino']
Curtosis_Nina=HistoNinaData.kurt()['Nina']
Curtosis_Neutro=HistoNeutroData.kurt()['Neutro']


      