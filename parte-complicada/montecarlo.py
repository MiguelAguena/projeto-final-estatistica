import pandas as pd
from pathlib import Path
import statistics as st
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import mplcursors

def normalMonteCarlo(avgs, std_dev, num_reps):
    i = 0
    arr = []
    while(i < len(avgs)):
        arr_i = np.random.normal(avgs[i], std_dev, num_reps)
        arr.append(arr_i)
        i += 1
    return arr

def calcZ(r):
    return 0.5 * np.log((1 + r)/(1 - r))

def calcR(z):
    return ((np.exp(2 * z) - 1) / (np.exp(2 * z) + 1))

def calcReturns(data):
    i = 0
    returns = []
    while(i < len(data) - 1):
        returns.append(np.log(data[i + 1] / data[i]))
        i = i + 1
  
    return returns

def metrics(weights, log_mean, covariance):
    weights = np.array(weights)
    returns = np.array([
        log_mean[0] * weights[0],
        log_mean[1] * weights[1],
        log_mean[2] * weights[2],
        log_mean[3] * weights[3]
    ])
    volatility = np.sqrt([
        covariance[0 ] * weights[0] * weights[0],
        covariance[1 ] * weights[0] * weights[1],
        covariance[2 ] * weights[0] * weights[2],
        covariance[3 ] * weights[0] * weights[3],
        covariance[4 ] * weights[1] * weights[0],
        covariance[5 ] * weights[1] * weights[1],
        covariance[6 ] * weights[1] * weights[2],
        covariance[7 ] * weights[1] * weights[3],
        covariance[8 ] * weights[2] * weights[0],
        covariance[9 ] * weights[2] * weights[1],
        covariance[10] * weights[2] * weights[2],
        covariance[11] * weights[2] * weights[3],
        covariance[12] * weights[3] * weights[0],
        covariance[13] * weights[3] * weights[1],
        covariance[14] * weights[3] * weights[2],
        covariance[15] * weights[3] * weights[3],
    ])
    sharpe_ratio = np.sum(returns) / np.sum(volatility)
    return [np.sum(returns), np.sum(volatility), sharpe_ratio]

def __main__():
    df = pd.read_csv(Path('assets.csv')).dropna()
    amount = len(df.index)

    petr = df['PETR4.SA'].array * 252
    wege = df['WEGE3.SA'].array * 252
    abev = df['ABEV3.SA'].array * 252
    vale = df['VALE3.SA'].array * 252

    ret_petr  = calcReturns(petr)
    ret_wege  = calcReturns(wege)
    ret_abev  = calcReturns(abev)
    ret_vale  = calcReturns(vale)

    ret_means = [st.mean(ret_petr), st.mean(ret_wege), st.mean(ret_abev), st.mean(ret_vale)]

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
    corr_wege_petr = st.correlation(ret_wege, ret_petr)
    corr_wege_abev = st.correlation(ret_wege, ret_abev)
    corr_wege_vale = st.correlation(ret_wege, ret_vale)
    corr_abev_petr = st.correlation(ret_abev, ret_petr)
    corr_abev_wege = st.correlation(ret_abev, ret_wege)
    corr_abev_vale = st.correlation(ret_abev, ret_vale)
    corr_vale_petr = st.correlation(ret_vale, ret_petr)
    corr_vale_wege = st.correlation(ret_vale, ret_wege)
    corr_vale_abev = st.correlation(ret_vale, ret_abev)

    correlations = [
        corr_petr_wege,
        corr_petr_abev,
        corr_petr_vale,
        corr_wege_petr,
        corr_wege_abev,
        corr_wege_vale,
        corr_abev_petr,
        corr_abev_wege,
        corr_abev_vale,
        corr_vale_petr,
        corr_vale_wege,
        corr_vale_abev
    ]

    print("Correlations")
    print(correlations)

    z_calc_all = calcZ(np.array(correlations))
    print("CALC z all")
    print(z_calc_all)

    z_desvpad = 1 / np.sqrt(amount - 3)
    print("Z desvpad")
    print(z_desvpad)

    sims = 100
    z_all = normalMonteCarlo(z_calc_all, z_desvpad, sims)
    print("z all")
    print(z_all)

    r_rand = calcR(z_all)
    print("Monte Carlo")
    print(r_rand)

    cov_petr_wege = r_rand[0 ] * np.sqrt(petr_var) * np.sqrt(wege_var)
    cov_petr_abev = r_rand[1 ] * np.sqrt(petr_var) * np.sqrt(abev_var)
    cov_petr_vale = r_rand[2 ] * np.sqrt(petr_var) * np.sqrt(vale_var)
    cov_wege_petr = r_rand[3 ] * np.sqrt(wege_var) * np.sqrt(petr_var)
    cov_wege_abev = r_rand[4 ] * np.sqrt(wege_var) * np.sqrt(abev_var)
    cov_wege_vale = r_rand[5 ] * np.sqrt(wege_var) * np.sqrt(vale_var)
    cov_abev_petr = r_rand[6 ] * np.sqrt(abev_var) * np.sqrt(petr_var)
    cov_abev_wege = r_rand[7 ] * np.sqrt(abev_var) * np.sqrt(wege_var)
    cov_abev_vale = r_rand[8 ] * np.sqrt(abev_var) * np.sqrt(vale_var)
    cov_vale_petr = r_rand[9 ] * np.sqrt(vale_var) * np.sqrt(petr_var)
    cov_vale_wege = r_rand[10] * np.sqrt(vale_var) * np.sqrt(wege_var)
    cov_vale_abev = r_rand[11] * np.sqrt(vale_var) * np.sqrt(abev_var)

    covs = [
        np.full(sims, petr_var),
        cov_petr_wege,
        cov_petr_abev,
        cov_petr_vale,
        np.full(sims, wege_var),
        cov_wege_petr,
        cov_wege_abev,
        cov_wege_vale,
        np.full(sims, abev_var),
        cov_abev_petr,
        cov_abev_wege,
        cov_abev_vale,
        np.full(sims, vale_var),
        cov_vale_petr,
        cov_vale_wege,
        cov_vale_abev,
    ]

    print("Covariances")
    print(covs)

    ## Markowitz
    print("#####################")
    i = 0
    while(i < 1):
        print("Exec: " + str(i))
        cov_to_test = [covs[0 ][i],
                       covs[1 ][i],
                       covs[2 ][i],
                       covs[3 ][i],
                       covs[4 ][i],
                       covs[5 ][i],
                       covs[6 ][i],
                       covs[7 ][i],
                       covs[8 ][i],
                       covs[9 ][i],
                       covs[10][i],
                       covs[11][i],
                       covs[12][i],
                       covs[13][i],
                       covs[14][i],
                       covs[15][i]]
        
        # Define the bounds for portfolio weights (between 0 and 1)
        bounds = [(0, 1)] * (len(df.columns) - 1)

        # Define an initial guess for portfolio weights (equal weights)
        initial_guess = [(1 / (len(df.columns) - 1))] * (len(df.columns) - 1)

        # Define an initial constraint that ensures the sum of weights equals 1
        constraints = [{'type':'eq','fun':lambda weights: np.sum(weights) - 1}]

        # Calculate weights and returns for the optimizations and input filtering
        maximum_return_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x
        maximum_return = metrics(maximum_return_weights, np.array(ret_means), np.array(cov_to_test))[0]
        
        minimum_risk_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[1], initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x
        minimum_risk = metrics(minimum_risk_weights, np.array(ret_means), np.array(cov_to_test))[1]
        minimum_risk_return = metrics(minimum_risk_weights, np.array(ret_means), np.array(cov_to_test))[0]
        
        maximum_risk_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[1] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x
        maximum_risk = metrics(maximum_risk_weights, np.array(ret_means), np.array(cov_to_test))[1]

        print("Maximum return: " + str(maximum_return))
        print("  Weights: " + str(maximum_return_weights))
        print("Minimum risk: " + str(minimum_risk))
        print("Maximum return of minimum risk: " + str(minimum_risk_return))
        print("  Weights: " + str(minimum_risk_weights))
        print("Maximum risk: " + str(maximum_risk))
        print("  Weights: " + str(maximum_risk_weights))

        sharpe_ratio_optimal_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[2] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x

        calculation_type = input("Calc type: ")
        if calculation_type == 'risk':
            risk_tolerance = input("Expected risk: ")
            # Find optimal weights that maximize return (minimize return * -1)
            constraints.append({'type': 'ineq', 'fun': lambda weights: risk_tolerance - metrics(weights, np.array(ret_means), np.array(cov_to_test))[1]})
            optimal_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x


        elif calculation_type == 'return':
            expected_return = input("Expected return: ")
            # Find optimal weights that minimize risk
            constraints.append({'type': 'eq', 'fun': lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] - float(expected_return)})
            optimal_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[1], initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x


        # Generate a range of target returns
        target_returns = np.linspace(minimum_risk_return, maximum_return, 100)

        efficient_frontier_volatility = []
        efficient_frontier_return = []

        # Calculate the efficient frontier
        for target_return in target_returns:
            # Optimize for minimum volatility given the target return
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                {'type': 'eq', 'fun': lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] - target_return}
            ]
            result = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[1], initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

            efficient_frontier_volatility.append(result.fun)
            efficient_frontier_return.append(target_return)


        #
        # Graph
        #

        plt.style.use('./mplstyles/financialgraphs.mplstyle')

        markowitz_optimization, axes = plt.subplots(figsize=(14, 8))

        axes.plot(efficient_frontier_volatility, efficient_frontier_return, label='Efficient Frontier')

        axes.scatter(metrics(sharpe_ratio_optimal_weights, np.array(ret_means), np.array(cov_to_test))[1], metrics(sharpe_ratio_optimal_weights, np.array(ret_means), np.array(cov_to_test))[0], marker='o', label='Max. Sharpe Ratio')
        axes.scatter(metrics(optimal_weights, np.array(ret_means), np.array(cov_to_test))[1], metrics(optimal_weights, np.array(ret_means), np.array(cov_to_test))[0], marker='o', label='Optimal Portfolio')

        axes.xaxis.set_major_formatter(mplticker.PercentFormatter(1.0))
        axes.yaxis.set_major_formatter(mplticker.PercentFormatter(1.0))
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        axes.set_title('Anual Expected Return x Volatility')

        legend_text = '\n'.join([f'{metric}: {metrics(optimal_weights, np.array(ret_means), np.array(cov_to_test))[j]:.2%}' for j, metric in enumerate(['Expected Return','Volatility','Sharpe Ratio'])]) + '\n\n'
        legend_text = legend_text + '\n'.join([f'{asset}\'s weight: {j:.2%}' for asset, j in zip(df.columns.tolist(), optimal_weights)]) + '\n'
        plt.legend(title=f'{legend_text}')

        # Enable cursor interaction on the graph
        cursor = mplcursors.cursor()
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.get_bbox_patch().set(fc='gray', alpha=0.8)
            sel.annotation.get_bbox_patch().set_edgecolor('gray')
            sel.annotation.arrow_patch.set_color('white')
            sel.annotation.arrow_patch.set_arrowstyle('-')

        plt.show()

        i += 1

__main__()