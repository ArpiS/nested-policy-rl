import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from scipy import optimize

class LMM():

	def __init__(self):
		pass

	def fit(self, X, y, groups, method="bfgs"):

		if method == "bfgs":

			def f(x):
				beta_shared, beta_fg = x
				preds = np.squeeze(X) * beta_shared + groups * np.squeeze(X) * beta_fg
				# MSE
				return np.mean((y - preds)**2)

			# Initial value of x
			x0 = np.random.normal(size=2)

			# Try with BFGS
			xopt = optimize.minimize(f,x0,method='bfgs',options={'disp':1})

			self.coefs_shared = xopt.x[0]
			self.coefs_fg = xopt.x[1]

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


	def predict(self, X, y, groups):
		pass


if __name__ == "__main__":

	# simple example
	n = 200
	p = 1
	coefs_shared_true = np.array([1])
	coefs_fg_true = np.array([4])
	X = np.random.normal(0, 1, size=(n, p))

	
	groups = np.random.binomial(n=1, p=0.5, size=n)

	# Shared effect
	y = X @ coefs_shared_true + np.random.normal(0, 1, n)

	# Foreground-specific effect
	y[groups == 1] = y[groups == 1] + X[groups == 1, :] @ coefs_fg_true

	# Fit LMM
	lmm = LMM()
	lmm.fit(X, y, groups=groups)

	# Plot
	    
	data = pd.DataFrame(X, columns=["X"])
	data['y'] = y
	data['group'] = groups
	sns.scatterplot(data=data, x="X", y="y", hue="group")
	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = 0 + lmm.coefs_shared * x_vals
	plt.plot(x_vals, y_vals, '--', label="Shared coef")

	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = 0 + (lmm.coefs_fg + lmm.coefs_shared) * x_vals
	plt.plot(x_vals, y_vals, '--', label="FG coef")
	plt.legend()

	plt.show()

	import ipdb; ipdb.set_trace()
