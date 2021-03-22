import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import pandas as pd
import seaborn as sns
from scipy import optimize, special
import ipdb
import pandas as pd
import ipdb


class LMM():

    def __init__(self, model='regression', num_classes=None, is_pendulum=True):

        self.model = model
        self.num_classes = num_classes
        self.is_pendulum = is_pendulum
        if self.model == 'classification' and self.num_classes == None:
            raise Exception("Need to specify number of classes if model is classification")

    def fit(self, X, y, groups, method="bfgs", verbose=True):
        import matplotlib.pyplot as plt
        n, p = X.shape
        if method == "bfgs":
            # Add columns of ones for intercept
            X = np.hstack([np.ones((n, 1)), X])

            def f(x):
                # optimize MSE
                if self.model == 'regression':
                    beta_shared, beta_fg = x[:p + 1], x[p + 1:]
                    assert beta_shared.shape[0] == beta_fg.shape[0]
                    preds = X @ beta_shared + (X @ beta_fg) * groups
                    mse = np.mean((y - preds) ** 2)
                    return mse
                elif self.model == 'classification':
                    # Reshape from flattened vector
                    x = np.reshape(x, [2 * p + 2, self.num_classes])
                    beta_shared, beta_fg = x[:p+1, :], x[p + 1:, :]
                    
                    # Linear function
                    preds = X @ beta_shared + (X @ beta_fg) * np.expand_dims(groups, 1)

                    # Logsistic function
                    preds = special.expit(preds)

                    # Compute cross entropy
                    if self.is_pendulum:
                        ce = log_loss(y, preds, labels=[-2, -1, 0, 1, 2])
                    else:
                        preds -= special.logsumexp(preds, axis=1)[:, np.newaxis]
                        loss = -(y * preds).sum()

                    return loss
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
            xopt = optimize.minimize(f, x0, method='bfgs', options={'disp': verbose})

            if self.model == 'classification':
                # Reshape from flattened vector
                # import ipdb; ipdb.set_trace()
                xstar = np.reshape(xopt.x, [2 * p + 2, self.num_classes])
                self.coefs_shared = xstar[:p + 1, :]
                self.coefs_fg = xstar[p + 1:, :]
            elif self.model == 'regression':
                self.coefs_shared = xopt.x[:p + 1]
                self.coefs_fg = xopt.x[p + 1:]

        # Not implemented for 12 dimensions
        elif method == "project":

            # Regression on all samples
            X_bg = X[groups == 0]
            y_bg = y[groups == 0]
            reg = LinearRegression().fit(X_bg, y_bg)
            all_coefs_shared = np.concatenate([[reg.intercept_], reg.coef_])
            coefs_shared = all_coefs_shared

            # Get residuals for foreground group
            X_fg = X[groups == 1]
            y_fg = y[groups == 1]
            X_fg_preds = reg.predict(X_fg)
            X_residuals = y_fg - X_fg_preds

            # Regress residuals on the foreground
            reg = LinearRegression().fit(X_fg, X_residuals)
            all_coefs_fg = np.concatenate([[reg.intercept_], reg.coef_])
            coefs_fg = all_coefs_fg

            self.coefs_shared = coefs_shared
            self.coefs_fg = coefs_fg

        else:
            raise Exception("Method must be one of [bfgs, project]")

    def predict(self, X, groups):
        # Add columns of ones for intercept
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])
        
        if self.model == 'regression':
            # only need linear part
            preds = X @ self.coefs_shared + (X @ self.coefs_fg) * groups

        elif self.model == 'classification':
            preds = X @ self.coefs_shared + (X @ self.coefs_fg) * np.expand_dims(groups, 1)
            preds = np.argmax(preds, axis=1)
            if self.is_pendulum:
                preds -= 2

        return preds


if __name__ == "__main__":
    pass


