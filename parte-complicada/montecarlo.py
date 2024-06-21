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
    returns = np.sum(np.array([
        log_mean[0] * weights[0],
        log_mean[1] * weights[1],
        log_mean[2] * weights[2],
        log_mean[3] * weights[3]
    ]))

    volatility = np.sqrt(weights.T.dot(covariance.dot(weights)))

    sharpe_ratio = returns / volatility
    return [returns, volatility, sharpe_ratio]

def __main__():
    df = pd.read_csv(Path('assets.csv')).dropna()
    amount = len(df.index)

    petr = df['PETR4.SA'].array
    wege = df['WEGE3.SA'].array
    abev = df['ABEV3.SA'].array
    vale = df['VALE3.SA'].array

    ret_petr  = np.delete(np.log(petr / petr.shift(1)), 0)
    ret_wege  = np.delete(np.log(wege / wege.shift(1)), 0)
    ret_abev  = np.delete(np.log(abev / abev.shift(1)), 0)
    ret_vale  = np.delete(np.log(vale / vale.shift(1)), 0)

    ret_means = np.array([st.mean(ret_petr), st.mean(ret_wege), st.mean(ret_abev), st.mean(ret_vale)]) * 252

    print("Return means")
    print(ret_means)

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
        cov_wege_petr,
        np.full(sims, wege_var),
        cov_wege_abev,
        cov_wege_vale,
        cov_abev_petr,
        cov_abev_wege,
        np.full(sims, abev_var),
        cov_abev_vale,
        cov_vale_petr,
        cov_vale_wege,
        cov_vale_abev,
        np.full(sims, vale_var),
    ]

    print("Covariances")
    print(covs)

    ## Markowitz
    print("#####################")

    calculation_type = input("Calculation type: ")
    expected_value = input("Expected variable value: ")

    linspace_min = 100
    linspace_max = 0

    total_efficient_frontier_volatility = []
    total_efficient_frontier_return = []

    i = 0
    while(i < 1):
        print("Exec: " + str(i))
        cov_to_test = [[covs[0 ][i],
                       covs[1 ][i],
                       covs[2 ][i],
                       covs[3 ][i]],
                       [covs[4 ][i],
                       covs[5 ][i],
                       covs[6 ][i],
                       covs[7 ][i]],
                       [covs[8 ][i],
                       covs[9 ][i],
                       covs[10][i],
                       covs[11][i]],
                       [covs[12][i],
                       covs[13][i],
                       covs[14][i],
                       covs[15][i]]]
        
        print("Return means")
        print(ret_means)

        print("Covs")
        print(cov_to_test)

        cov_to_test = np.array(cov_to_test) * 252

        print("Covs * 252")
        print(cov_to_test)

        # Define the bounds for portfolio weights (between 0 and 1)
        bounds = [(0, 1)] * (len(df.columns) - 1)

        # Define an initial guess for portfolio weights (equal weights)
        initial_guess = [(1 / (len(df.columns) - 1))] * (len(df.columns) - 1)

        # Define an initial constraint that ensures the sum of weights equals 1
        constraints = [{'type':'eq','fun':lambda weights: np.sum(weights) - 1}]

        # Calculate weights and returns for the optimizations and input filtering
        maximum_return_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints,).x
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

        
        if calculation_type == 'risk':
            risk_tolerance = expected_value
            # Find optimal weights that maximize return (minimize return * -1)
            constraints.append({'type': 'ineq', 'fun': lambda weights: float(risk_tolerance) - metrics(weights, np.array(ret_means), np.array(cov_to_test))[1]})
            optimal_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] * -1, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x
            print(optimal_weights)
            print(metrics(optimal_weights, np.array(ret_means), np.array(cov_to_test)))


        elif calculation_type == 'return':
            expected_return = expected_value
            # Find optimal weights that minimize risk
            constraints.append({'type': 'eq', 'fun': lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[0] - float(expected_return)})
            optimal_weights = optimize.minimize(lambda weights: metrics(weights, np.array(ret_means), np.array(cov_to_test))[1], initial_guess, method='SLSQP', bounds=bounds, constraints=constraints).x
            print(optimal_weights)
            print(metrics(optimal_weights, np.array(ret_means), np.array(cov_to_test)))

        if(linspace_min > minimum_risk_return) :
            linspace_min = minimum_risk_return

        if(linspace_max < maximum_return) :
            linspace_max = maximum_return

        # Generate a range of target returns
        target_returns = np.linspace(linspace_min, linspace_max, 100)

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

        total_efficient_frontier_volatility.append(efficient_frontier_volatility)
        total_efficient_frontier_return.append(efficient_frontier_return)

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
        legend_text = legend_text + '\n'.join([f'{asset}\'s weight: {j:.2%}' for asset, j in zip(['PETR4', 'WEGE3', 'ABEV3', 'VALE3'], optimal_weights)]) + '\n'
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
    
    i = 0

__main__()