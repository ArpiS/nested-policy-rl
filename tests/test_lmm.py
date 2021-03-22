import sys
sys.path.append("../simulated_fqi/models/")

from lmm import LMM
import numpy as np
from scipy.special import expit

def test_lmm_classification():

    for _ in range(20):
        # simple example
        n = 100
        p = 1
        k = 2 # number of classes
        coefs_shared_true = np.random.normal(size=(p+1, k))
        coefs_fg_true = np.random.normal(size=(p+1, k))
        X = np.random.normal(0, 1, size=(n, p))

        groups = np.random.binomial(n=1, p=0.5, size=n)

        # Add columns of ones for intercept
        X_ext = np.hstack([np.ones((n, 1)), X])
        y = X_ext @ coefs_shared_true + (X_ext @ coefs_fg_true) * np.expand_dims(groups, 1)
        y = np.argmax(y, axis=1)
        y_onehot = np.zeros((n, k))
        for k_idx in range(k):
            y_onehot[y == k_idx, k_idx] = 1


        # Fit LMM
        lmm = LMM(model='classification', num_classes=k, is_pendulum=False)
        lmm.fit(X, y_onehot, groups=groups, verbose=False)

        # Make sure most of predictions match
        preds = lmm.predict(X, groups=groups)
        assert np.mean(y == preds) > 0.9

def test_lmm_regression():

    for _ in range(20):

        n = 200
        p = 10
        coefs_shared_true = np.repeat([1], p + 1)
        coefs_fg_true = np.repeat([4], p + 1)
        X = np.random.normal(0, 1, size=(n, p))

        groups = np.random.binomial(n=1, p=0.5, size=n)

        # Add columns of ones for intercept
        X_ext = np.hstack([np.ones((n, 1)), X])
        y = X_ext @ coefs_shared_true + (X_ext @ coefs_fg_true) * groups

        # Fit LMM
        lmm = LMM(model='regression', is_pendulum=False)
        lmm.fit(X, y, groups=groups, verbose=False)

        assert np.allclose(lmm.coefs_shared, coefs_shared_true, atol=1e-4)
        assert np.allclose(lmm.coefs_fg, coefs_fg_true, atol=1e-4)
        
        fitted_values = lmm.predict(X, groups=groups)
        
        assert np.allclose(fitted_values, y, atol=1e-4)

def test_lmm_regression_project():

    for _ in range(20):

        n = 200
        p = 10
        coefs_shared_true = np.random.normal(size=(p+1))
        coefs_fg_true = np.random.normal(size=(p+1))
        X = np.random.normal(0, 1, size=(n, p))

        groups = np.random.binomial(n=1, p=0.5, size=n)

        # Add columns of ones for intercept
        X_ext = np.hstack([np.ones((n, 1)), X])
        y = X_ext @ coefs_shared_true + (X_ext @ coefs_fg_true) * groups

        # Fit LMM
        lmm = LMM(model='regression', is_pendulum=False)
        lmm.fit(X, y, groups=groups, verbose=False, method="project")

        assert np.allclose(lmm.coefs_shared, coefs_shared_true, atol=1e-4)
        assert np.allclose(lmm.coefs_fg, coefs_fg_true, atol=1e-4)
        
        fitted_values = lmm.predict(X, groups=groups)
        
        assert np.allclose(fitted_values, y, atol=1e-4)

if __name__ == "__main__":
    test_lmm_regression_project()
    test_lmm_classification()
    test_lmm_regression()
    



