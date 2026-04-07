import pandas as pd
import numpy as np
days=['_day_1.csv','_day_2.csv','_day_3.csv']
ImportTariffArr=[]
ExportTariffArr=[]
SunlightArr=[]
TransportArr=[]
SugarArr=[]
PrisMidArr=[]
MAGNIFICENT_MACARONS='MAGNIFICENT_MACARONS'
totalrow=[]
totaly=[]
for day in days:
    ObservationDF=pd.read_csv('Level4\\round-4-island-data-bottle\\observations_round_4' + day,sep=',')
    PricesDF=pd.read_csv('Level4\\round-4-island-data-bottle\\prices_round_4' + day,sep=';')
    PricesDF=PricesDF[PricesDF['product']==MAGNIFICENT_MACARONS]
    controlFactors=['transportFees','exportTariff','importTariff','sugarPrice','sunlightIndex']
    calcMatrix=[]
    yMatrix=[]

    for time in PricesDF['timestamp']:
        ObservationTimeFrame=ObservationDF[ObservationDF['timestamp']==time]
        PristineMidPrice=np.mean([ObservationTimeFrame['bidPrice'],ObservationTimeFrame['askPrice']])
        row=[]  
        for factor in controlFactors:
            row.append(ObservationTimeFrame[factor].iloc[0])
        row.append(PristineMidPrice)
        # calcMatrix.append(np.array(row))
        totalrow.append(np.array(row))
        # yMatrix.append(PricesDF[PricesDF['timestamp']==time]['mid_price'].iloc[0])
        totaly.append(PricesDF[PricesDF['timestamp']==time]['mid_price'].iloc[0])
        # if len(yMatrix)==5:
        #     print(calcMatrix)
        #     print(np.shape(calcMatrix))
        #     print(np.shape(yMatrix))
        #     T,E,I,Sug,Sun=np.linalg.solve(calcMatrix,yMatrix)
        #     TransportArr.append(T)
        #     ExportTariffArr.append(E)
        #     ImportTariffArr.append(I)
        #     SugarArr.append(Sug)
        #     SunlightArr.append(Sun)
        #     calcMatrix=[]
        #     yMatrix=[]
testingwindow=1000000
for run in range(testingwindow):
    yMatrix=[]
    calcMatrix=[]
    for i in range(6):
        randomnum=int(np.random.rand()*(len(totalrow)-1))
        yMatrix.append(totaly[randomnum])
        calcMatrix.append(totalrow[randomnum])
    if np.linalg.matrix_rank(calcMatrix) < len(controlFactors)+1:
        print("Singular matrix detected. Skipping this batch.")
        calcMatrix = []
        yMatrix = []
        continue
    T,E,I,Sug,Sun,PrisMid=np.linalg.solve(calcMatrix,yMatrix)
    TransportArr.append(T)
    ExportTariffArr.append(E)
    ImportTariffArr.append(I)
    SugarArr.append(Sug)
    SunlightArr.append(Sun)   
    PrisMidArr.append(PrisMid)         
    calcMatrix = []
    yMatrix = []
    print('run '+str(run))
print('sugar mean: '+ str(np.mean(SugarArr)))
print('sugar stdev: '+ str(np.sqrt(np.var(SugarArr))))
print('export mean: '+ str(np.mean(ExportTariffArr)))
print('export stdev: '+ str(np.sqrt(np.var(ExportTariffArr))))
print('import mean: '+ str(np.mean(ImportTariffArr)))
print('import stdev: '+ str(np.sqrt(np.var(ImportTariffArr))))
print('transport mean: '+ str(np.mean(TransportArr)))
print('transport stdev: '+ str(np.sqrt(np.var(TransportArr))))
print('sun mean: '+ str(np.mean(SunlightArr)))
print('sun stdev: '+ str(np.sqrt(np.var(SunlightArr))))
print('PrisMid mean: '+ str(np.mean(PrisMidArr)))
print('PrisMid stdev: '+ str(np.sqrt(np.var(PrisMidArr))))