import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(covariance_matrix, industry_classification, company_identifiers):
    n = covariance_matrix.shape[0]

    def objective(x):
        return np.sqrt(np.dot(x.T, np.dot(covariance_matrix, x)))

    def constraint1(x):
        return np.sum(x) - 1.0

    def constraint2(x):
        industries = industry_classification['Industry'].unique()
        for industry in industries:
            mask = industry_classification['Industry'] == industry
            if np.sum(x[mask]) > 0.25:
                return False
        return True

    def constraint3(x):
        return np.all(x >= 0) and np.all(x <= 0.08)

    constraints = [{'type': 'eq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2},
                   {'type': 'ineq', 'fun': constraint3}]

    x0 = np.ones(n) / n
    result = minimize(objective, x0, constraints=constraints)

    optimal_weights = result.x

    df = pd.DataFrame({'Industry': industry_classification['Industry'],
                       'Optimal Weight': optimal_weights}, index=company_identifiers)
    return df


#  covariance_matrix: a NxN pandas dataframe representing the covariance matrix of the stocks in the portfolio

# industry_classification: a Nx1 pandas dataframe containing the industry classification of each stock, with a company identifier in the Index and an "Industry" column containing the industry classification

# company_identifiers: a list of N company identifiers, corresponding to the rows in the covariance matrix and the industry classification dataframe

import numpy as np
import pandas as pd

# Example data for the covariance matrix
covariance_matrix = pd.DataFrame({'Stock 1': [0.03, 0.02, 0.01],
                                   'Stock 2': [0.02, 0.05, 0.03],
                                   'Stock 3': [0.01, 0.03, 0.04]},
                                  index=['Stock 1', 'Stock 2', 'Stock 3'])

# Example data for the industry classification
industry_classification = pd.DataFrame({'Industry': ['Technology', 'Finance', 'Finance']},
                                        index=['Stock 1', 'Stock 2', 'Stock 3'])

# Example list of company identifiers
company_identifiers = ['Stock 1', 'Stock 2', 'Stock 3']

# Call the optimize_portfolio function
result = optimize_portfolio(covariance_matrix, industry_classification, company_identifiers)
print(result)
