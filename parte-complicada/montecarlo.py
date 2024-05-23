import pandas as pd
from pathlib import Path
import statistics as st
import numpy as np

def normalMonteCarlo(avg, std_dev, num_reps):
    i = 0
    arr = np.random.normal(avg, std_dev, num_reps)
    return arr

def calcZ(corr):
    return 0.5 * np.log((1 + corr)/(1 - corr))

def calcReturns(data):
    i = 0
    returns = []
    while(i < len(data) - 1):
        returns.append(np.log(data[i + 1] / data[i]))
        i = i + 1
  
    return returns

def __main__():
    df = pd.read_csv(Path('assets.csv')).dropna()
    amount = len(df.index)

    petr = df['PETR4.SA'].array
    wege = df['WEGE3.SA'].array
    abev = df['ABEV3.SA'].array
    vale = df['VALE3.SA'].array

    ret_petr  = calcReturns(petr)
    ret_wege  = calcReturns(wege)
    ret_abev  = calcReturns(abev)
    ret_vale  = calcReturns(vale)

    petr_var = st.variance(ret_petr)
    wege_var = st.variance(ret_wege)
    abev_var = st.variance(ret_abev)
    vale_var = st.variance(ret_vale)

    print("Variances")
    print(petr_var)
    print(wege_var)
    print(abev_var)
    print(vale_var)

    corr_petr_wege = st.correlation(ret_petr, ret_wege)
    corr_petr_abev = st.correlation(ret_petr, ret_abev)
    corr_petr_vale = st.correlation(ret_petr, ret_vale)
    corr_wege_abev = st.correlation(ret_wege, ret_abev)
    corr_wege_vale = st.correlation(ret_wege, ret_vale)
    corr_abev_vale = st.correlation(ret_abev, ret_vale)

    print("Correlations")
    print(corr_petr_wege)
    print(corr_petr_abev)
    print(corr_petr_vale)
    print(corr_wege_abev)
    print(corr_wege_vale)
    print(corr_abev_vale)

    z_petr_wege = calcZ(corr_petr_wege)
    z_petr_abev = calcZ(corr_petr_abev)
    z_petr_vale = calcZ(corr_petr_vale)
    z_wege_abev = calcZ(corr_wege_abev)
    z_wege_vale = calcZ(corr_wege_vale)
    z_abev_vale = calcZ(corr_abev_vale)

    print("Zs")
    print(z_petr_wege)
    print(z_petr_abev)
    print(z_petr_vale)
    print(z_wege_abev)
    print(z_wege_vale)
    print(z_abev_vale)

    z_desvpad = 1 / np.sqrt(amount - 3)
    print("Z desvpad")
    print(z_desvpad)

    sims = 100
    print("Monte Carlo")
    print(normalMonteCarlo(z_petr_wege, z_desvpad, sims))
    print(normalMonteCarlo(z_petr_abev, z_desvpad, sims))
    print(normalMonteCarlo(z_petr_vale, z_desvpad, sims))
    print(normalMonteCarlo(z_wege_abev, z_desvpad, sims))
    print(normalMonteCarlo(z_wege_vale, z_desvpad, sims))
    print(normalMonteCarlo(z_abev_vale, z_desvpad, sims))

__main__()