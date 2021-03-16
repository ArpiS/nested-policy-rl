import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import pandas as pd
import seaborn as sns
from scipy import optimize, special
import ipdb
import torch
import torch.nn as nn
import pandas as pd


class LMM():

    def __init__(self, model='regression', num_classes=None):
        self.model = model
        self.num_classes = num_classes
        if self.model == 'classification' and self.num_classes == None:
            raise Exception("Need to specify number of classes if model is classification")

    def fit(self, X, y, groups, method="bfgs"):

        n, p = X.shape
        if method == "bfgs":
            # Add columns of ones for intercept
            X = np.hstack([np.ones((n, 1)), X])

            def f(x):
                # optimize MSE
                if self.model == 'regression':
                    beta_shared, beta_fg = x[:p + 1], x[p + 1:]
                    preds = X @ beta_shared + np.multiply(groups, X) @ beta_fg
                    return np.mean((y - preds) ** 2)
                elif self.model == 'classification':
                    
                    # Reshape from flattened vector
                    x = np.reshape(x, [2 * p + 2, self.num_classes])
                    beta_shared, beta_fg = x[:p+1, :], x[p + 1:, :]
                    
                    # Linear function
                    preds = X @ beta_shared + np.multiply(groups, X) @ beta_fg

                    # Logsistic function
                    preds = special.expit(preds)
                    # print(preds)

                    # Normalize each row to sum to 1 (ie, be a valid prob. distribution)
                    preds = preds / (preds.sum(axis=1) + 1e-4)[:,None]

                    # Compute cross entropy
                    ce = log_loss(y, preds)

                    return ce
                else:
                    raise Exception("Model must be either regression or classification")


            # Initial value of x
            # (need 2 times the params to account for both groups)
            # (separate row of params for each class)
            if self.model == 'classification':
                x0 = np.random.normal(size=(2 * p + 2, self.num_classes))
                # Need to flatten for optimizer
                x0 = np.ndarray.flatten(x0)
            else:
                x0 = np.random.normal(size=2 * p + 2)

            # Try with BFGS
            xopt = optimize.minimize(f, x0, method='bfgs', options={'disp': 1})

            # Reshape from flattened vector
            xstar = np.reshape(xopt.x, [2 * p + 2, self.num_classes])
            self.coefs_shared = xstar[:p + 1, :]
            self.coefs_fg = xstar[p + 1:, :]

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

        if self.model == 'regression':
            # only need linear part
            pass

        elif self.model == 'classification':
            preds = np.argmax(preds, axis=1)

        return preds


if __name__ == "__main__":

    # Classification example
    n = 200
    p = 5
    k = 3
    lmm = LMM(model="classification", num_classes=k)

    coefs_shared_true = np.random.normal(size=(p, k))
    coefs_fg_true = np.random.normal(size=(p, k))
    X = np.random.normal(size=(n, p))
    groups = np.zeros(n)
    groups[n//2:] = 1
    groups = np.expand_dims(groups, 1)

    preds = X @ coefs_shared_true + np.multiply(groups, X) @ coefs_fg_true

    y = np.argmax(preds, axis=1)
    
    lmm.fit(X, y, groups=groups)

    Xtest = np.random.normal(size=(n, p))
    groups = np.zeros(n)
    groups[n//2:] = 1
    groups = np.expand_dims(groups, 1)

    preds = Xtest @ coefs_shared_true + np.multiply(groups, Xtest) @ coefs_fg_true

    ytest = np.argmax(preds, axis=1)

    yhat = lmm.predict(Xtest, groups)
    acc = np.mean(ytest == yhat)
    print("accuracy: {}".format(round(acc, 3)))
    import ipdb; ipdb.set_trace()

    


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