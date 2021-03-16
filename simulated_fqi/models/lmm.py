import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import pandas as pd
import seaborn as sns
from scipy import optimize, special


class LMM():

    def __init__(self):
        pass

    def fit(self, X, y, groups, method="bfgs", model='regression'):

        n, p = X.shape
        if method == "bfgs":

            # Add columns of ones for intercept
            X = np.hstack([np.ones((n, 1)), X])
            
            def f(x):
                beta_shared, beta_fg = x[:p + 1], x[p + 1:]
                preds = X @ beta_shared + np.multiply(groups, X) @ beta_fg
                # optimize MSE
                if model == 'regression':
                    return np.mean((y - preds) ** 2)
                elif model == 'classification':
                    preds = special.expit(preds)
                    print("PREDS: ", preds)
                    return log_loss(y, preds)
                else:
                    raise Exception("Model must be either regression or classification")


            # Initial value of x
            # (need 2 times the params to account for both groups)
            x0 = np.random.normal(size=2 * p + 2)
            
            # Try with BFGS
            xopt = optimize.minimize(f, x0, method='bfgs', options={'disp': 0})
            #import ipdb; ipdb.set_trace()
            self.coefs_shared = xopt.x[:p + 1]
            self.coefs_fg = xopt.x[p + 1:]

        # Not implemented for 12 dimensions
        elif method == "project":

            # Regression on all samples
            reg = LinearRegression().fit(X, y)
            coefs_shared = reg.coef_

            # Get residuals for foreground group
            X_fg = X[groups == 1]
            y_fg = y[groups == 1]
            X_fg_preds = reg.predict(X_fg)
            X_residuals = y_fg - X_fg_preds

            # Regress residuals on the foreground
            reg = LinearRegression().fit(X_fg, X_residuals)
            coefs_fg = reg.coef_

            self.coefs_shared = coefs_shared
            self.coefs_fg = coefs_fg

        else:
            raise Exception("Method must be one of [bfgs, project]")

    def predict(self, X, groups):
        # Add columns of ones for intercept
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])

        # Shared part + fg-specific part
        preds = X @ self.coefs_shared + np.multiply(groups, X) @ self.coefs_fg
        return preds


if __name__ == "__main__":
    # simple example
    n = 200
    p = 1
    coefs_shared_true = np.repeat([1], p + 1)
    coefs_shared_true = np.reshape(coefs_shared_true, (p + 1, 1))
    coefs_fg_true = np.repeat([4], p + 1)
    coefs_fg_true = np.reshape(coefs_fg_true, (p + 1, 1))
    X = np.random.normal(0, 1, size=(n, p))

    groups = np.random.binomial(n=1, p=0.5, size=n)
    groups = np.expand_dims(groups, 1)

    # Shared effect

    # Add columns of ones for intercept
    X_ext = np.hstack([np.ones((n, 1)), X])
    y = X_ext @ coefs_shared_true + np.multiply(groups, X_ext) @ coefs_fg_true
    # noise
    y = np.squeeze(y) + np.random.normal(0, 1, n)

    y = np.random.randint(0, 2, y.shape)

    # Fit LMM
    lmm = LMM()
    lmm.fit(X, y, groups=groups, model='classification')

    print(np.allclose(lmm.coefs_shared, coefs_shared_true, rtol=0.2))
    print(np.allclose(lmm.coefs_fg, coefs_fg_true, rtol=0.2))

    # Test on a random test set
    X_test = np.random.normal(0, 1, size=(n, p))
    X_test_ext = np.hstack([np.ones((n, 1)), X_test])
    y_test = X_test_ext @ coefs_shared_true + np.multiply(groups, X_test_ext) @ coefs_fg_true
    # noise
    y_test = np.squeeze(y_test) + np.random.normal(0, 1, n)
    y_test = np.random.randint(0, 2, y_test.shape)

    groups_test = np.random.binomial(n=1, p=0.5, size=n)
    groups_test = np.expand_dims(groups_test, axis=1)

    preds = lmm.predict(X_test, groups_test)