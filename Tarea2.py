# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:31:06 2021

@author: dinue
"""

import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import scipy
from sympy import symbols, solve, gamma
from scipy.special import psi 

data=pd.read_csv('CAUDAL_H@26057040.data',sep="|")
data['Fecha'] = pd.to_datetime(data['Fecha'])
#Cambiar indice por fecha
data = data.set_index('Fecha')

#Agrupar por fechas y calcular maximos
MaxDailyData = data.groupby(pd.Grouper(freq='1D')).max()

#Agrupar por fechas y calcular minimos
MinDailyData = data.groupby(pd.Grouper(freq='1D')).mean()

#Restablecer indices
MaxDailyData.reset_index(inplace=True)
MinDailyData.reset_index(inplace=True)

#Exportar archivo con datos diarios promedio
MaxDailyData.to_csv('Maximos.data',index=False)
MinDailyData.to_csv('Minimos.data',index=False)

#Importarlos de nuevo:
MaxDailyData2=pd.read_csv('Maximos.data',sep=",")
MinDailyData2=pd.read_csv('Minimos.data',sep=",")

##Punto 1: Hallar caudales maximos anuales
#Para tener los rangos para caudales maximos
indexes=[0]
for i in range(len(MaxDailyData2)):
    if MaxDailyData2['Fecha'][i][5:10]=='06-01':
        indexes.append(i)
indexes.append(len(MaxDailyData2)-1)
        
#Lista de valores segun año
DailyMaxDischarges=[]
for i in range(0,len(MaxDailyData2)):
    DailyMaxDischarges.append(MaxDailyData2['Valor'][i])

#Caudales maximos        
MaxDischarges=[]
for i in range(0,len(indexes)-1):
    MaxDischarges.append(max(DailyMaxDischarges[indexes[i]:indexes[i+1]]))
    
#Grafica de la serie de Qmax

TicksPlot1=[1976]
for i in range(1980,2021,10):
    TicksPlot1.append(i)

# # Gráfica de la serie
# plt.close('all')
# plt.figure('Qmax')
# plt.plot(range(1976,2021),MaxDischarges)
# plt.grid(True)
# plt.title('Serie de caudales máximos anuales')
# plt.xlabel('Año')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.xticks(TicksPlot1)

#DataFrame para calcular estadísticos
QmaxDF=pd.DataFrame(MaxDischarges,columns=['Valor'])
QmaxDF.to_csv('MaxMax.csv')
#Media
MeanMax=QmaxDF.mean()['Valor']

#Desviacion Estandar
StandardDeviationMax=QmaxDF.std()['Valor']

#Asimetría
SkewnessMax=QmaxDF.skew()['Valor']

#AsimetriaAlternativa
M3=[]
for i in range(0,len(MaxDischarges)):
    M3.append((MaxDischarges[i]-MeanMax)**3)

M3T=sum(M3)

SkAlt=M3T/(len(MaxDischarges)*StandardDeviationMax**3)
#Kurtosis
KurtosisMax=QmaxDF.kurt()['Valor']

# #Histograma de Qmax
# QmaxDF.hist(cumulative=False,density=1,bins=20)
# plt.title('Histograma de frecuencias')
# plt.xlabel('Caudal ($m^3/s$)')

# QmaxDF.hist(cumulative=True,density=1,bins=45)
# plt.title('Función de distribución de probabilidad acumulada')
# plt.xlabel('Caudal ($m^3/s$)')

#Punto 2: Caudales segun periodo de retorno y distribución

Tr=[2.33,5,10,25,50,100,500]

p=[]
for i in range(0,len(Tr)):
    p.append(1/(Tr[i]))
    
q=[]
for i in range(0,len(p)):
    q.append(1-p[i])

ProbDF=pd.DataFrame(list(zip(Tr,p,q)),columns=['Tr','p','q'])
ProbDF.to_csv('Probs.csv')
    
#Distribucion de Gumbel

#Parametro alpha
alphaG=math.sqrt(6)*(StandardDeviationMax)/(math.pi)
betaG=MeanMax-0.5772*(alphaG)
xTG=[]
for i in range(0,len(q)):
    xTG.append(betaG-alphaG*math.log(-math.log(q[i])))
KTG=[]
for i in range(0,len(Tr)):
    KTG.append(-math.sqrt(6)/math.pi*(0.5772+math.log(math.log(Tr[i]/(Tr[i]-1)))))
    
#Test Smirnov-Kolmogorov

#Ordenando la lista de Qmax
MaxDischarges2=[]
for i in range(0,len(MaxDischarges)):
    MaxDischarges2.append(MaxDischarges[i])

MaxDischarges2.sort()

#PDF Empirica
EDF1=[]
for i in range (1,len(MaxDischarges2)+1):
    EDF1.append(i/len(MaxDischarges2))

EDF2=[]    
for i in range (1,len(MaxDischarges2)+1):
    EDF2.append((i-1)/len(MaxDischarges2))

CritStat=1.36/np.sqrt(len(MaxDischarges2))

#CDF de Gumbel para la muestra
cdf_gumbmax_hyp=[]
for i in range (0,len(MaxDischarges2)):
    cdf_gumbmax_hyp.append(math.exp(-math.exp(-((MaxDischarges2[i]-betaG)/alphaG))))

#Diferencia en valor absoluto
#EDF1
AbsDifEDF1G=[]
for i in range(0,len(EDF1)):
    AbsDifEDF1G.append(np.abs(EDF1[i]-cdf_gumbmax_hyp[i]))
AbsDifEDF1Max=max(AbsDifEDF1G)    

#EDF2
AbsDifEDF2G=[]
for i in range(0,len(EDF2)):
    AbsDifEDF2G.append(np.abs(EDF2[i]-cdf_gumbmax_hyp[i]))
AbsDifEDF2Max=max(AbsDifEDF2G)  

AbsDifMaxG=max(AbsDifEDF1Max,AbsDifEDF2Max)

if AbsDifMaxG<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para Gumbel')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para Gumbel')

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. de Gumbel')
# plt.plot(MaxDischarges2,EDF1,label='$F_N(x)$')
# plt.plot(MaxDischarges2,cdf_gumbmax_hyp,label='Dist. de Gumbel')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. de Gumbel')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(65,0.93,'$D_n=0.1084$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')

#Intervalo de confianza: xT+-z_{a/2}*ST
mu_1G=betaG+0.5772*alphaG
mu_2G=math.pi**2*alphaG**2/6
gamma_1G=1.1396
gamma_2G=5.4
STG=[]
for i in range(0,len(KTG)):
    STG.append(math.sqrt(mu_2G/len(MaxDischarges)*(1+KTG[i]*gamma_1G+KTG[i]**2/4*(gamma_2G-1))))
z_0025=1.96    
LowerLimitG=[]
for i in range(0,len(STG)):
    LowerLimitG.append(xTG[i]-z_0025*STG[i])

UpperLimitG=[]
for i in range(0,len(STG)):
    UpperLimitG.append(xTG[i]+z_0025*STG[i])
    
ConfIntG=[]
for i in range(0,len(LowerLimitG)):
    ConfIntG.append([LowerLimitG[i],UpperLimitG[i]])

# #Grafica Intervalos de Confianza
# plt.figure('Intervalos de Confianza Dist. de Gumbel')
# plt.plot(Tr,LowerLimitG,'--',marker='o',label='Límite inferior')
# plt.plot(Tr,xTG,marker='o',label='Qmax')
# plt.plot(Tr,UpperLimitG,'--',marker='o',label='Límite superior')
# plt.grid(True)
# plt.title('Intervalos de Confianza Dist. de Gumbel')
# plt.xscale('log')
# plt.xlabel('Periodo de retorno (años)')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.legend(loc='lower right')

    
#Distribucion Log-Normal
CVX=StandardDeviationMax/MeanMax
Sigma_YLN=math.sqrt(math.log(CVX**2+1))
Mu_YLN=math.log(MeanMax)-Sigma_YLN**2/2    
znorm=[]
for i in range(0,len(q)):
    znorm.append(st.norm.ppf(q[i]))
xTLN=[]
for i in range(0,len(q)):
    xTLN.append(math.exp(Mu_YLN+Sigma_YLN*znorm[i]))

KTLN=[]
for i in range(0,len(Tr)):
    KTLN.append((xTLN[i]-MeanMax)/StandardDeviationMax)

KTLNAlt=[]
for i in range(0,len(Tr)):
    KTLNAlt.append((xTLN[i]-math.exp(Mu_YLN+Sigma_YLN**2/2))/math.sqrt((math.exp(Sigma_YLN**2)-1)*math.exp(2*Mu_YLN+Sigma_YLN**2)))    
    
#Con funcion built-in de Python
xTLNAlt=[]
for i in range(0,len(q)):
    xTLNAlt.append(st.lognorm(Sigma_YLN,scale=np.exp(Mu_YLN)).ppf(q[i]))
    
#Test Smirnov-Kolmogorov

#CDF Log-Normal para la muestra
cdf_lognorm_hyp=[]
for i in range(0,len(MaxDischarges2)):
    cdf_lognorm_hyp.append(st.norm.cdf((math.log(MaxDischarges2[i])-Mu_YLN)/Sigma_YLN))

#Diferencia en valor absoluto
#EDF1
AbsDifEDF1LN=[]
for i in range(0,len(EDF1)):
    AbsDifEDF1LN.append(np.abs(EDF1[i]-cdf_lognorm_hyp[i]))
AbsDifEDF1MaxLN=max(AbsDifEDF1LN)    

#EDF2
AbsDifEDF2LN=[]
for i in range(0,len(EDF2)):
    AbsDifEDF2LN.append(np.abs(EDF2[i]-cdf_lognorm_hyp[i]))
AbsDifEDF2MaxLN=max(AbsDifEDF2LN)  

AbsDifMaxLN=max(AbsDifEDF1MaxLN,AbsDifEDF2MaxLN)    

if AbsDifMaxLN<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para Lognormal')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para Lognormal')

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. Log-Normal')
# plt.plot(MaxDischarges2,EDF1,label='$F_N(x)$')
# plt.plot(MaxDischarges2,cdf_lognorm_hyp,label='Dist. Log-Normal')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. Log-Normal')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(65,0.93,'$D_n=0.0987$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')

#Intervalo de confianza: xT+-z_{a/2}*ST
mu_1LN=math.exp(Mu_YLN+Sigma_YLN**2/2)
mu_2LN=(math.exp(Sigma_YLN**2)-1)*math.exp(2*Mu_YLN+Sigma_YLN**2)
gamma_1LN=(math.exp(Sigma_YLN**2)+2)*math.sqrt(math.exp(Sigma_YLN**2)-1)
gamma_2LN=math.exp(4*Sigma_YLN**2)+2*math.exp(3*Sigma_YLN**2)+3*math.exp(2*Sigma_YLN**2)-3

STLN=[]
for i in range(0,len(KTLN)):
    STLN.append(math.sqrt(mu_2LN/len(MaxDischarges)*(1+KTLN[i]*gamma_1LN+KTLN[i]**2/4*(gamma_2LN-1))))
    
LowerLimitLN=[]
for i in range(0,len(STLN)):
    LowerLimitLN.append(xTLN[i]-z_0025*STLN[i])

UpperLimitLN=[]
for i in range(0,len(STLN)):
    UpperLimitLN.append(xTLN[i]+z_0025*STLN[i])
    
ConfIntLN=[]
for i in range(0,len(LowerLimitLN)):
    ConfIntLN.append([LowerLimitLN[i],UpperLimitLN[i]])

# #Grafica Intervalos de Confianza
# plt.figure('Intervalos de Confianza Dist. Log-Normal')
# plt.plot(Tr,LowerLimitLN,'--',marker='o',label='Límite inferior')
# plt.plot(Tr,xTLN,marker='o',label='Qmax')
# plt.plot(Tr,UpperLimitLN,'--',marker='o',label='Límite superior')
# plt.grid(True)
# plt.title('Intervalos de Confianza Dist. Log-Normal')
# plt.xscale('log')
# plt.xlabel('Periodo de retorno (años)')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.legend(loc='lower right')


#Distribucion de Frechet

#Log de la muestra de Qmax
lambdaFrech2=6.219687009567408579809835 #De Mathematica #=beta 
tauFrech2=MeanMax/gamma(1-1/lambdaFrech2) #=theta

MeanFrech2=tauFrech2*gamma(1-1/lambdaFrech2)
STDFrech2=tauFrech2*math.sqrt(gamma(1-2/lambdaFrech2)-gamma(1-1/lambdaFrech2)**2)


xTFrech=[]
for i in range(0,len(Tr)):
    xTFrech.append(tauFrech2*(math.log(Tr[i]/(Tr[i]-1)))**(-1/lambdaFrech2))

KTFrech=[]
for i in range(0,len(Tr)):
    KTFrech.append((xTFrech[i]-MeanMax)/StandardDeviationMax)
    
#Test Smirnov-Kolmogorov

#CDF Frechet para la muestra
cdf_frechet_hyp=[]
for i in range(0,len(MaxDischarges2)):
    cdf_frechet_hyp.append(math.exp(-(tauFrech2/MaxDischarges2[i])**(lambdaFrech2)))

#Diferencia en valor absoluto
#EDF1
AbsDifEDF1Frech=[]
for i in range(0,len(EDF1)):
    AbsDifEDF1Frech.append(np.abs(EDF1[i]-cdf_frechet_hyp[i]))
AbsDifEDF1MaxFrech=max(AbsDifEDF1Frech)    

#EDF2
AbsDifEDF2Frech=[]
for i in range(0,len(EDF2)):
    AbsDifEDF2Frech.append(np.abs(EDF2[i]-cdf_frechet_hyp[i]))
AbsDifEDF2MaxFrech=max(AbsDifEDF2Frech)  

AbsDifMaxFrech=max(AbsDifEDF1MaxFrech,AbsDifEDF2MaxFrech)

if AbsDifMaxFrech<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para Frechet')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para Frechet')

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. de Frechet')
# plt.plot(MaxDischarges2,EDF1,label='$F_N(x)$')
# plt.plot(MaxDischarges2,cdf_frechet_hyp,label='Dist. de Frechet')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. de Frechet')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(65,0.93,'$D_n=0.1356$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')

#Intervalo de confianza: xT+-z_{a/2}*ST
def DrF(r):
    return gamma(1-r/lambdaFrech2)

mu_1Frech=MeanFrech2
mu_2Frech=STDFrech2**2
mu_3Frech=tauFrech2**3*(DrF(3)-3*DrF(2)*DrF(1)+2*DrF(1)**3)
mu_4Frech=tauFrech2**4*(DrF(4)-4*DrF(3)*DrF(1)+6*DrF(2)*DrF(1)**2-3*DrF(1)**4)

gamma_1Frech=mu_3Frech/mu_2Frech**(3/2)
gamma_2Frech=mu_4Frech/mu_2Frech**(2)

STFrech=[]
for i in range(0,len(KTFrech)):
    STFrech.append(math.sqrt(mu_2Frech/len(MaxDischarges)*(1+KTFrech[i]*gamma_1Frech+KTFrech[i]**2/4*(gamma_2Frech-1))))
    
LowerLimitFrech=[]
for i in range(0,len(STFrech)):
    LowerLimitFrech.append(xTFrech[i]-z_0025*STFrech[i])

UpperLimitFrech=[]
for i in range(0,len(STFrech)):
    UpperLimitFrech.append(xTFrech[i]+z_0025*STFrech[i])
    
ConfIntFrech=[]
for i in range(0,len(LowerLimitFrech)):
    ConfIntFrech.append([LowerLimitFrech[i],UpperLimitFrech[i]])        

# #Grafica Intervalos de Confianza
# plt.figure('Intervalos de Confianza Dist. de Frechet')
# plt.plot(Tr,LowerLimitFrech,'--',marker='o',label='Límite inferior')
# plt.plot(Tr,xTFrech,marker='o',label='Qmax')
# plt.plot(Tr,UpperLimitFrech,'--',marker='o',label='Límite superior')
# plt.grid(True)
# plt.title('Intervalos de Confianza Dist. de Frechet')
# plt.xscale('log')
# plt.xlabel('Periodo de retorno (años)')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.legend(loc='lower right')

#Distribucion de Weibull
betaWMax=-0.729268-0.338679*SkewnessMax+4.96077*(SkewnessMax+1.14)**(-1.0422)+0.683609*(math.log(SkewnessMax+1.14))**2
alphaWMax=StandardDeviationMax/math.sqrt(gamma(1+2/betaWMax)-(gamma(1+1/betaWMax))**2)
xiWMax=MeanMax-alphaWMax*gamma(1+1/betaWMax)

xTWMax=[]
for i in range(0,len(Tr)):
    xTWMax.append(xiWMax+alphaWMax*(-math.log(1/Tr[i]))**(1/betaWMax))
    
KTWMax=[]
for i in range(0,len(Tr)):
    KTWMax.append((xTWMax[i]-MeanMax)/StandardDeviationMax)
    
#Test Smirnov-Kolmogorov
    
#CDF Log-Normal para la muestra
cdf_weibullmax_hyp=[]
for i in range(0,len(MaxDischarges2)):
    cdf_weibullmax_hyp.append(1-math.exp(-((MaxDischarges2[i]-xiWMax)/alphaWMax)**(betaWMax)))

#Diferencia en valor absoluto
#EDF1
AbsDifEDF1WMax=[]
for i in range(0,len(EDF1)):
    AbsDifEDF1WMax.append(np.abs(EDF1[i]-cdf_weibullmax_hyp[i]))
AbsDifEDF1MaxWMax=max(AbsDifEDF1WMax)    

#EDF2
AbsDifEDF2WMax=[]
for i in range(0,len(EDF2)):
    AbsDifEDF2WMax.append(np.abs(EDF2[i]-cdf_weibullmax_hyp[i]))
AbsDifEDF2MaxWMax=max(AbsDifEDF2WMax)  

AbsDifMaxWMax=max(AbsDifEDF1MaxWMax,AbsDifEDF2MaxWMax)    

if AbsDifMaxWMax<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para WeibullMax')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para WeibullMax')

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. de Weibull')
# plt.plot(MaxDischarges2,EDF1,label='$F_N(x)$')
# plt.plot(MaxDischarges2,cdf_weibullmax_hyp,label='Dist. de Weibull')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. de Weibull')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(65,0.93,'$D_n=0.0880$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')

#Intervalo de confianza: xT+-z_{a/2}*ST

def DrWM(r):
    return gamma(1+r/betaWMax)

def psirWM(r):
    return psi(1+r/betaWMax)

mu_2WMax=alphaWMax**2*(DrWM(2)-DrWM(1)**2)
mu_3WMax=alphaWMax**3*(DrWM(3)-3*DrWM(2)*DrWM(1)+2*DrWM(1)**3)
mu_4WMax=alphaWMax**4*(DrWM(4)-4*DrWM(3)*DrWM(1)+6*DrWM(2)*DrWM(1)**2-3*DrWM(1)**4)
mu_5WMax=alphaWMax**5*(DrWM(5)-5*DrWM(4)*DrWM(1)+10*DrWM(3)*(DrWM(1)**2)-10*DrWM(2)*(DrWM(1)**2)+4*DrWM(1)**5)
mu_6WMax=alphaWMax**6*(DrWM(6)-6*DrWM(5)*DrWM(1)+15*DrWM(4)*DrWM(1)**2-20*DrWM(3)*DrWM(1)**3+15*DrWM(2)*DrWM(1)**4-5*DrWM(1)**6)
gamma_1WMax=mu_3WMax/mu_2WMax**(3/2)
gamma_2WMax=mu_4WMax/mu_2WMax**(2)
gamma_3WMax=mu_5WMax/mu_2WMax**(5/2)
gamma_4WMax=mu_6WMax/mu_2WMax**(3)

dKTWMax=[]
dKT1WMax=[]
BKTWMax=[]
for i in range(0,len(Tr)):
    BKTWMax.append(math.log(Tr[i]))

#psi=digamma
for i in range(0,len(Tr)):
    dKT1WMax.append(((DrWM(1)*psirWM(1)-math.log(BKTWMax[i])*BKTWMax[i]**(1/betaWMax))*(DrWM(2)-DrWM(1)**2)-(BKTWMax[i]**(1/betaWMax)-DrWM(1))*(-DrWM(2)*psirWM(2)+DrWM(1)**2*psirWM(1)))/(betaWMax**2*(DrWM(2)-DrWM(1)**2)**(3/2)))

dKT2WMax=(betaWMax**2*(DrWM(2)-DrWM(1)**2)**(5/2))/(3*((DrWM(2)-DrWM(1)**2)*(-DrWM(3)*psirWM(3)+2*DrWM(2)*DrWM(1)*psirWM(2)+DrWM(2)*DrWM(1)*psirWM(1)-2*DrWM(1)**3*psirWM(1))-(DrWM(3)-3*DrWM(2)*DrWM(1)+2*DrWM(1)**3)*(-DrWM(2)*psirWM(2)+DrWM(1)**2*psirWM(1))))

for i in range(0,len(Tr)):
    dKTWMax.append(dKT1WMax[i]/dKT2WMax)

STWMaxT1=[]
for i in range(0,len(KTWMax)):
    STWMaxT1.append(dKTWMax[i]*(2*gamma_2WMax-3*gamma_1WMax**2-6+KTWMax[i]*(gamma_3WMax-6*gamma_1WMax*gamma_2WMax/4-10*gamma_1WMax/4)))

STWMaxT2=[]
for i in range(0,len(KTWMax)):
    STWMaxT2.append(dKTWMax[i]**2*(gamma_4WMax-3*gamma_1WMax*gamma_3WMax-6*gamma_2WMax+9*gamma_1WMax**2*gamma_2WMax/4+35*gamma_1WMax**2/4+9))    

STWMax=[]
for i in range(0,len(STWMaxT1)):
    STWMax.append(math.sqrt(mu_2WMax/len(MaxDischarges)*(1+KTWMax[i]*gamma_1WMax+KTWMax[i]**2/4*(gamma_2WMax-1))))

# STWMax=[]
# for i in range(0,len(KTWMax)):
#     STWMax.append(mu_2WMax/len(MaxDischarges)*(1+KTWMax[i]*gamma_1WMax+KTWMax[i]**2/4*(gamma_2WMax-1)+dKTWMax[i]*(2*gamma_2WMax-3*gamma_1WMax**2-6+KTWMax[i]*(gamma_3WMax-6*gamma_1WMax*gamma_2WMax/4-10*gamma_1WMax/4))+dKTWMax[i]**2*(gamma_4WMax-3*gamma_1WMax*gamma_3WMax-6*gamma_2WMax-9*gamma_1WMax**2*gamma_2WMax/4+35*gamma_1WMax**2/4+9)))
    
LowerLimitWMax=[]
for i in range(0,len(STWMax)):
    LowerLimitWMax.append(xTWMax[i]-z_0025*STWMax[i])

UpperLimitWMax=[]
for i in range(0,len(STWMax)):
    UpperLimitWMax.append(xTWMax[i]+z_0025*STWMax[i])
    
ConfIntWMax=[]
for i in range(0,len(LowerLimitWMax)):
    ConfIntWMax.append([LowerLimitWMax[i],UpperLimitWMax[i]])

# #Grafica Intervalos de Confianza
# plt.figure('Intervalos de Confianza Dist. de Weibull')
# plt.plot(Tr,LowerLimitWMax,'--',marker='o',label='Límite inferior')
# plt.plot(Tr,xTWMax,marker='o',label='Qmax')
# plt.plot(Tr,UpperLimitWMax,'--',marker='o',label='Límite superior')
# plt.grid(True)
# plt.title('Intervalos de Confianza Dist. de Weibull')
# plt.xscale('log')
# plt.xlabel('Periodo de retorno (años)')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.legend(loc='lower right')

# Distribucion de Pareto
xiPar=-0.6312563794 #De Mathematica
sigmaPar=StandardDeviationMax*(1-xiPar)*math.sqrt(1-2*xiPar)
muPar=MeanMax-sigmaPar/(1-xiPar)

xTPar=[]
for i in range(0,len(q)):
    xTPar.append(muPar-sigmaPar/xiPar*(1-(1-q[i])**(-xiPar)))

KTPar=[]
for i in range(0,len(Tr)):
    KTPar.append((xTPar[i]-MeanMax)/StandardDeviationMax)

#Test Smirnov-Kolmogorov   
cdf_pareto_hyp=[]
for i in range(2,len(MaxDischarges2)):
    cdf_pareto_hyp.append(1-(1+xiPar/sigmaPar*(MaxDischarges2[i]-muPar))**(-1/xiPar))
    
#EDF1
AbsDifEDF1Par=[]
for i in range(0,len(cdf_pareto_hyp)):
    AbsDifEDF1Par.append(np.abs(EDF1[i+2]-cdf_pareto_hyp[i]))
AbsDifEDF1MaxPar=max(AbsDifEDF1Par)    

#EDF2
AbsDifEDF2Par=[]
for i in range(0,len(cdf_pareto_hyp)):
    AbsDifEDF2Par.append(np.abs(EDF2[i+2]-cdf_pareto_hyp[i]))
AbsDifEDF2MaxPar=max(AbsDifEDF2Par)  

AbsDifMaxPar=max(AbsDifEDF1MaxPar,AbsDifEDF2MaxPar)    

if AbsDifMaxPar<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para Pareto')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para Pareto') 

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. Generalizada de Pareto')
# plt.plot(MaxDischarges2[2:len(MaxDischarges2)],EDF1[2:len(MaxDischarges2)],label='$F_N(x)$')
# plt.plot(MaxDischarges2[2:len(MaxDischarges2)],cdf_pareto_hyp,label='Dist. Gen. de Pareto')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. Generalizada de Pareto')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(80,0.93,'$D_n=0.0902$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')
    
#Intervalo de confianza: xT+-z_{a/2}*ST
mu_1P=muPar+sigmaPar/(1-xiPar)
mu_2P=sigmaPar**2/((1-xiPar)**2*(1-2*xiPar))

gamma_1P=(2*(1+xiPar)*math.sqrt(1-2*xiPar))/(1-3*xiPar)
gamma_2P=(3*(1-2*xiPar)*(2*xiPar**2+xiPar+3))/((1-3*xiPar)*(1-4*xiPar))

STPar=[]
for i in range(0,len(KTPar)):
    STPar.append(math.sqrt(mu_2P/len(MaxDischarges)*(1+KTPar[i]*gamma_1P+KTPar[i]**2/4*(gamma_2P-1))))
    
LowerLimitPar=[]
for i in range(0,len(STPar)):
    LowerLimitPar.append(xTPar[i]-z_0025*STPar[i])

UpperLimitPar=[]
for i in range(0,len(STPar)):
    UpperLimitPar.append(xTPar[i]+z_0025*STPar[i])
    
ConfIntPar=[]
for i in range(0,len(LowerLimitPar)):
    ConfIntPar.append([LowerLimitPar[i],UpperLimitPar[i]])

#Exportando los resultados en un DataFrame    
xTDF=pd.DataFrame(list(zip(Tr,p,q,xTG,xTLN,xTFrech,xTWMax,xTPar)),columns=['Tr','p','q','Gumbel','Log-Normal','Frechet','Weibul','Pareto'])
xTDF.to_csv('Qdiseno.csv')

#CI Gumbel
CIGDF=pd.DataFrame(list(zip(Tr,KTG,STG,LowerLimitG,xTG,UpperLimitG)),columns=['Tr','KT','ST','Limite inferior','Qmax','Limite superior'])
CIGDF.to_csv('QGumbel.csv')

#CI LogNormal
CILNDF=pd.DataFrame(list(zip(Tr,KTLN,STLN,LowerLimitLN,xTLN,UpperLimitLN)),columns=['Tr','KT','ST','Limite inferior','Qmax','Limite superior'])
CILNDF.to_csv('QLogNormal.csv')

#CI Frechet
CIFrechDF=pd.DataFrame(list(zip(Tr,KTFrech,STFrech,LowerLimitFrech,xTFrech,UpperLimitFrech)),columns=['Tr','KT','ST','Limite inferior','Qmax','Limite superior'])
CIFrechDF.to_csv('QFrechet.csv')

#CI Weibull
CIWMaxDF=pd.DataFrame(list(zip(Tr,KTWMax,STWMax,LowerLimitWMax,xTWMax,UpperLimitWMax)),columns=['Tr','KT','ST','Limite inferior','Qmax','Limite superior'])
CIWMaxDF.to_csv('QWeibullMax.csv')

#CI Pareto
CIParDF=pd.DataFrame(list(zip(Tr,KTPar,STPar,LowerLimitPar,xTPar,UpperLimitPar)),columns=['Tr','KT','ST','Limite inferior','Qmax','Limite superior'])
CIParDF.to_csv('QPareto.csv')

# #Grafica Intervalos de Confianza
# plt.figure('Intervalos de Confianza Dist. Gen. de Pareto')
# plt.plot(Tr,LowerLimitPar,'--',marker='o',label='Límite inferior')
# plt.plot(Tr,xTPar,marker='o',label='Qmax')
# plt.plot(Tr,UpperLimitPar,'--',marker='o',label='Límite superior')
# plt.grid(True)
# plt.title('Intervalos de Confianza Dist. Gen. de Pareto')
# plt.xscale('log')
# plt.xlabel('Periodo de retorno (años)')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.legend(loc='lower right')

#Punto 3: Caudales minimos

#Medias moviles de 7 dias

MovAve=[]
for i in range (0, len(MinDailyData)-6):
    MovAve.append(np.mean(MinDailyData['Valor'][i:i+7]))

for i in range(len(MinDailyData)-6,len(MinDailyData)):
    MovAve.append(199999999)
    
MinDailyData['Media movil']=MovAve    

#Lista de valores segun año
DailyMinDischarges=[]
for i in range(0,len(MinDailyData)):
    DailyMinDischarges.append(MinDailyData['Media movil'][i])

#Punto 3: Caudales minimos        
MinDischarges=[]
for i in range(0,len(indexes)-1):
    MinDischarges.append(min(DailyMinDischarges[indexes[i]:indexes[i+1]]))

# # Gráfica de la serie
# # plt.close('all')
# plt.figure('Qmin')
# plt.plot(range(1976,2021),MinDischarges)
# plt.grid(True)
# plt.title('Serie de caudales mínimos anuales')
# plt.xlabel('Año')
# plt.ylabel('Caudal ($m^3/s$)')
# plt.xticks(TicksPlot1)


#DataFrame para calcular estadísticos
QminDF=pd.DataFrame(MinDischarges,columns=['Valor'])
QminDF.to_csv('MinMin.csv')
#Media
MeanMin=QminDF.mean()['Valor']

#Desviacion Estandar
StandardDeviationMin=QminDF.std()['Valor']

#Asimetría
SkewnessMin=QminDF.skew()['Valor']

#Kurtosis
KurtosisMin=QminDF.kurt()['Valor']

# #Histograma de Qmin
# QminDF.hist(cumulative=False,density=1,bins=15)
# plt.title('Histograma de frecuencias')
# plt.xlabel('Caudal ($m^3/s$)')

# QminDF.hist(cumulative=True,density=1,bins=45)
# plt.title('Función de distribución de probabilidad acumulada')
# plt.xlabel('Caudal ($m^3/s$)')
    
#Distribucion de Weibull (xT=xbar-KT*S)
betaWMin=-0.729268-0.338679*SkewnessMin+4.96077*(SkewnessMin+1.14)**(-1.0422)+0.683609*(math.log(SkewnessMin+1.14))**2
alphaWMin=StandardDeviationMin/math.sqrt(gamma(1+2/betaWMin)-(gamma(1+1/betaWMin))**2)
xiWMin=MeanMin-alphaWMin*gamma(1+1/betaWMin)

xTWMin=[]
for i in range(1,len(Tr)-2):
    xTWMin.append(xiWMin+alphaWMin*(-math.log(1-1/Tr[i]))**(1/betaWMin))

KTWMin=[]
for i in range(0,len(xTWMin)):
    KTWMin.append((MeanMin-xTWMin[i])/StandardDeviationMin)    

#Test Smirnov-Kolmogorov

#Ordenando la lista de Qmin
MinDischarges2=[]
for i in range(0,len(MinDischarges)):
    MinDischarges2.append(MinDischarges[i])

MinDischarges2.sort()

#CDF Log-Normal para la muestra
cdf_weibullmin_hyp=[]
for i in range(0,len(MinDischarges2)):
    cdf_weibullmin_hyp.append(1-(1-math.exp(-((MinDischarges2[i]-xiWMin)/alphaWMin)**(betaWMin))))

#PDF Empirica
EDF1Min=[]
for i in range (1,len(MinDischarges2)+1):
    EDF1Min.append(1-i/len(MinDischarges2))

EDF2Min=[]    
for i in range (1,len(MinDischarges2)+1):
    EDF2Min.append(1-(i-1)/len(MinDischarges2))

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. de Weibull')
# plt.plot(MinDischarges2,EDF1Min,label='$F_N(x)$')
# plt.plot(MinDischarges2,cdf_weibullmin_hyp,label='Dist. de Weibull')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. de Weibull')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X>x)')
# plt.text(0.9,0.1,'$D_n=0.0935$',fontsize=15,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='upper right')


# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. de Weibull')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X<x)')
# plt.text(65,0.93,'$D_n=0.0880$',fontsize=20,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='lower right')
#Diferencia en valor absoluto
#EDF1
AbsDifEDF1WMin=[]
for i in range(0,len(EDF1Min)):
    AbsDifEDF1WMin.append(np.abs(EDF1Min[i]-cdf_weibullmin_hyp[i]))
AbsDifEDF1MaxWMin=max(AbsDifEDF1WMin)    

#EDF2
AbsDifEDF2WMin=[]
for i in range(0,len(EDF2Min)):
    AbsDifEDF2WMin.append(np.abs(EDF2Min[i]-cdf_weibullmin_hyp[i]))
AbsDifEDF2MaxWMin=max(AbsDifEDF2WMin)  

AbsDifMaxWMin=max(AbsDifEDF1MaxWMin,AbsDifEDF2MaxWMin)    

if AbsDifMaxWMin<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para WeibullMin')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para WeibullMin')
    
#Intervalo de confianza: xT+-z_{a/2}*ST


def DrWm(r):
    return gamma(1+r/betaWMin)

def psirWm(r):
    return psi(1+r/betaWMin)

mu_2WMin=alphaWMin**2*(DrWm(2)-DrWm(1)**2)
mu_3WMin=alphaWMin**3*(DrWm(3)-3*DrWm(2)*DrWm(1)+2*DrWm(1)**3)
mu_4WMin=alphaWMin**4*(DrWm(4)-4*DrWm(3)*DrWm(1)+6*DrWm(2)*DrWm(1)**2-3*DrWm(1)**4)
mu_5WMin=alphaWMin**5*(DrWm(5)-5*DrWm(4)*DrWm(1)+10*DrWm(3)*(DrWm(1)**2)-10*DrWm(2)*(DrWm(1)**2)+4*DrWm(1)**5)
mu_6WMin=alphaWMin**6*(DrWm(6)-6*DrWm(5)*DrWm(1)+15*DrWm(4)*DrWm(1)**2-20*DrWm(3)*DrWm(1)**3+15*DrWm(2)*DrWm(1)**4-5*DrWm(1)**6)
gamma_1WMin=mu_3WMin/mu_2WMin**(3/2)
gamma_2WMin=mu_4WMin/mu_2WMin**(2)
gamma_3WMin=mu_5WMin/mu_2WMin**(5/2)
gamma_4WMin=mu_6WMin/mu_2WMin**(3)

dKTWMin=[]
dKT1WMin=[]
BKTWMin=[]
for i in range(0,len(Tr)):
    BKTWMin.append(math.log(Tr[i]))

#psi=digamma
for i in range(0,len(Tr)):
    dKT1WMin.append(((DrWm(1)*psirWm(1)-math.log(BKTWMin[i])*BKTWMin[i]**(1/betaWMin))*(DrWm(2)-DrWm(1)**2)-(BKTWMin[i]**(1/betaWMin)-DrWm(1))*(-DrWm(2)*psirWm(2)+DrWm(1)**2*psirWm(1)))/(betaWMin**2*(DrWm(2)-DrWm(1)**2)**(3/2)))

dKT2WMin=(betaWMin**2*(DrWm(2)-DrWm(1)**2)**(5/2))/(3*((DrWm(2)-DrWm(1)**2)*(-DrWm(3)*psirWm(3)+2*DrWm(2)*DrWm(1)*psirWm(2)+DrWm(2)*DrWm(1)*psirWm(1)-2*DrWm(1)**3*psirWm(1))-(DrWm(3)-3*DrWm(2)*DrWm(1)+2*DrWm(1)**3)*(-DrWm(2)*psirWm(2)+DrWm(1)**2*psirWm(1))))

for i in range(0,len(Tr)):
    dKTWMin.append(dKT1WMin[i]/dKT2WMin)

STWMinT1=[]
for i in range(0,len(KTWMin)):
    STWMinT1.append(dKTWMin[i]*(2*gamma_2WMin-3*gamma_1WMin**2-6+KTWMin[i]*(gamma_3WMin-6*gamma_1WMin*gamma_2WMin/4-10*gamma_1WMin/4)))

STWMinT2=[]
for i in range(0,len(KTWMin)):
    STWMinT2.append(dKTWMin[i]**2*(gamma_4WMin-3*gamma_1WMin*gamma_3WMin-6*gamma_2WMin+9*gamma_1WMin**2*gamma_2WMin/4+35*gamma_1WMin**2/4+9))    

STWMin=[]
for i in range(0,len(STWMinT1)):
    STWMin.append(math.sqrt(mu_2WMin/len(MinDischarges)*(1+KTWMin[i]*gamma_1WMin+KTWMin[i]**2/4*(gamma_2WMin-1))))

# STWMin=[]
# for i in range(0,len(KTWMin)):
#     STWMin.append(math.sqrt(mu_2WMin/len(MinDischarges)*(1+KTWMin[i]*gamma_1WMin+KTWMin[i]**2/4*(gamma_2WMin-1)+STWMinT1[i]+STWMinT2[i])))
    
LowerLimitWMin=[]
for i in range(0,len(STWMin)):
    LowerLimitWMin.append(xTWMin[i]-z_0025*STWMin[i])

UpperLimitWMin=[]
for i in range(0,len(STWMin)):
    UpperLimitWMin.append(xTWMin[i]+z_0025*STWMin[i])
    
ConfIntWMin=[]
for i in range(0,len(LowerLimitWMin)):
    ConfIntWMin.append([LowerLimitWMin[i],UpperLimitWMin[i]])

#Grafica Intervalos de Confianza
plt.figure('Intervalos de Confianza Dist. de Weibull')
plt.plot(Tr[1:3],LowerLimitWMin[0:2],'--',marker='o',label='Límite inferior')
plt.plot(Tr[1:(len(q)-2)],xTWMin,marker='o',label='Qmin')
plt.plot(Tr[1:(len(q)-2)],UpperLimitWMin,'--',marker='o',label='Límite superior')
plt.grid(True)
plt.title('Intervalos de Confianza Dist. de Weibull')
plt.xlabel('Periodo de retorno (años)')
plt.ylabel('Caudal ($m^3/s$)')
plt.yticks(range(0,5))
plt.legend(loc='upper right')
    

#Distribucion Log-Normal
CVXMin=StandardDeviationMin/MeanMin
Sigma_YLNMin=math.sqrt(math.log(CVXMin**2+1))
Mu_YLNMin=math.log(MeanMin)-Sigma_YLNMin**2/2
    
znormMin=[]
for i in range(1,len(p)-2):
    znormMin.append(st.norm.ppf(p[i]))
xTLNMin=[]
for i in range(0,len(znormMin)):
    xTLNMin.append(math.exp(Mu_YLNMin+Sigma_YLNMin*znormMin[i]))

KTLNMin=[]
for i in range(0,len(znormMin)):
    KTLNMin.append((MeanMin-xTLNMin[i])/StandardDeviationMin)

# KTLNAlt=[]
# for i in range(0,len(Tr)):
#     KTLNAlt.append((xTLN[i]-math.exp(Mu_YLN+Sigma_YLN**2/2))/math.sqrt((math.exp(Sigma_YLN**2)-1)*math.exp(2*Mu_YLN+Sigma_YLN**2)))    
    
# #Con funcion built-in de Python
# xTLNAlt=[]
# for i in range(0,len(q)):
#     xTLNAlt.append(st.lognorm(Sigma_YLN,scale=np.exp(Mu_YLN)).ppf(q[i]))
    
# #Test Smirnov-Kolmogorov

#CDF Log-NormalMin para la muestra
cdf_lognormmin_hyp=[]
for i in range(0,len(MinDischarges2)):
    cdf_lognormmin_hyp.append(1-st.norm.cdf((math.log(MinDischarges2[i])-Mu_YLNMin)/Sigma_YLNMin))

# #Grafica EDF vs Distribucion
# plt.figure('$F_N(x)$ vs. Dist. LogNormalMin')
# plt.plot(MinDischarges2,EDF1Min,label='$F_N(x)$')
# plt.plot(MinDischarges2,cdf_lognormmin_hyp,label='Dist. Log-Normal')
# plt.grid(True)
# plt.title('$F_N(x)$ vs. Dist. Log-Normal')
# plt.xlabel('Caudal ($m^3/s$)')
# plt.ylabel('P(X>x)')
# plt.text(1,0.1,'$D_n=0.1054$',fontsize=15,style='italic',bbox={'facecolor': 'w', 'alpha':1, 'pad':10})
# plt.legend(loc='upper right')


#Diferencia en valor absoluto
#EDF1
AbsDifEDF1LNMin=[]
for i in range(0,len(EDF1Min)):
    AbsDifEDF1LNMin.append(np.abs(EDF1Min[i]-cdf_lognormmin_hyp[i]))
AbsDifEDF1MaxLNMin=max(AbsDifEDF1LNMin)    

#EDF2
AbsDifEDF2LNMin=[]
for i in range(0,len(EDF2Min)):
    AbsDifEDF2LNMin.append(np.abs(EDF2Min[i]-cdf_lognormmin_hyp[i]))
AbsDifEDF2MaxLNMin=max(AbsDifEDF2LNMin)  

AbsDifMaxLNMin=max(AbsDifEDF1MaxLNMin,AbsDifEDF2MaxLNMin)    

if AbsDifMaxLN<CritStat:
    print('La muestra pasa el test de Smirnov-Kolmogorov para LognormalMin')
else:
    print('La muestra NO pasa el test de Smirnov-Kolmogorov para LognormalMin')

#Intervalo de confianza: xT+-z_{a/2}*ST
mu_1LNMin=math.exp(Mu_YLNMin+Sigma_YLNMin**2/2)
mu_2LNMin=(math.exp(Sigma_YLNMin**2)-1)*math.exp(2*Mu_YLNMin+Sigma_YLNMin**2)
gamma_1LNMin=(math.exp(Sigma_YLNMin**2)+2)*math.sqrt(math.exp(Sigma_YLNMin**2)-1)
gamma_2LNMin=math.exp(4*Sigma_YLNMin**2)+2*math.exp(3*Sigma_YLNMin**2)+3*math.exp(2*Sigma_YLNMin**2)-3

STLNMin=[]
for i in range(0,len(KTLNMin)):
    STLNMin.append(math.sqrt(mu_2LNMin/len(MinDischarges)*(1+KTLNMin[i]*gamma_1LNMin+KTLNMin[i]**2/4*(gamma_2LNMin-1))))
    
LowerLimitLNMin=[]
for i in range(0,len(STLNMin)):
    LowerLimitLNMin.append(xTLNMin[i]-z_0025*STLNMin[i])

UpperLimitLNMin=[]
for i in range(0,len(STLNMin)):
    UpperLimitLNMin.append(xTLNMin[i]+z_0025*STLNMin[i])
    
ConfIntLNMin=[]
for i in range(0,len(LowerLimitLNMin)):
    ConfIntLNMin.append([LowerLimitLNMin[i],UpperLimitLNMin[i]])

#Grafica Intervalos de Confianza
plt.figure('Intervalos de Confianza Dist. Log-Normal')
plt.plot(Tr[1:3],LowerLimitLNMin[0:2],'--',marker='o',label='Límite inferior')
plt.plot(Tr[1:(len(q)-2)],xTLNMin,marker='o',label='Qmin')
plt.plot(Tr[1:(len(q)-2)],UpperLimitLNMin,'--',marker='o',label='Límite superior')
plt.grid(True)
plt.title('Intervalos de Confianza Dist. Log-Normal')
plt.xlabel('Periodo de retorno (años)')
plt.ylabel('Caudal ($m^3/s$)')
plt.yticks(range(0,6))
plt.legend(loc='lower right')


#Exportando los resultados en un DataFrame    
xTMinDF=pd.DataFrame(list(zip(Tr[1:(len(q)-2)],q[1:(len(q)-2)],p[1:(len(q)-2)],xTWMin,xTLNMin)),columns=['Tr','p','q','Weibull','Log-Normal'])
xTMinDF.to_csv('QdisenoMin.csv')

#CI Weibull
CIWMinDF=pd.DataFrame(list(zip(Tr[1:(len(q)-2)],KTWMin,STWMin,LowerLimitWMin,xTWMin,UpperLimitWMin)),columns=['Tr','KT','ST','Limite inferior','Qmin','Limite superior'])
CIWMinDF.to_csv('QWeibullMin.csv')

#CI LogNormal
CILNMinDF=pd.DataFrame(list(zip(Tr[1:(len(q)-2)],KTLNMin,STLNMin,LowerLimitLNMin,xTLNMin,UpperLimitLNMin)),columns=['Tr','KT','ST','Limite inferior','Qmin','Limite superior'])
CILNMinDF.to_csv('QLogNormalMin.csv')
    