import pandas as pd
from pathlib import Path
import statistics as st
import numpy as np

def NormalMonteCarlo(avg, std_dev, num_reps, sims):
    i = 0
    arr = []
    while(i < sims):
        arr.append(np.random.normal(avg, std_dev, num_reps))
        i = i + 1
    
    return arr

def calcZ(corr):
    return 0.5 * np.log((1 + corr)/(1 - corr))

def __main__():
    df = pd.read_csv(Path('assets.csv')).dropna()
    amount = len(df.index)

    petr = df['PETR4.SA']
    wege = df['WEGE3.SA']
    abev = df['ABEV3.SA']
    vale = df['VALE3.SA']

    petr_var = st.variance(petr)
    wege_var = st.variance(wege)
    abev_var = st.variance(abev)
    vale_var = st.variance(vale)

    corr_petr_wege = st.correlation(petr, wege)
    corr_petr_abev = st.correlation(petr, abev)
    corr_petr_vale = st.correlation(petr, vale)
    corr_wege_abev = st.correlation(wege, abev)
    corr_wege_vale = st.correlation(wege, vale)
    corr_abev_vale = st.correlation(abev, vale)

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

    print("Monte Carlo for PETR - WEGE")
    print(NormalMonteCarlo(z_petr_wege, z_desvpad, amount, 100))

__main__()