import numpy as np
from scipy import optimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def test_linear_model():

	n = 200
	p = 10
	X = np.random.normal(size=(n, p))
	beta = np.random.normal(size=p)
	y = X @ beta

	def f(param):
		preds = X @ param
		mse = np.mean((y - preds)**2)
		return mse

	x0 = np.random.normal(size=p)
	xopt = optimize.minimize(f, x0, method='bfgs', options={'disp': 1})
	assert np.allclose(xopt.x, beta, atol=1e-4)

def test_logistic_model():

	for _ in range(20):
		n = 200
		p = 2
		k = 3
		X = np.random.normal(size=(n, p))
		beta = np.random.normal(size=(p, k))
		y = X @ beta
		y = np.argmax(y, axis=1)

		lr = LogisticRegression()
		lr.fit(X, y)
		assert np.mean(lr.predict(X) == y) > 0.9
	

def test_contrastive_model():

	n = 200
	p = 10
	X = np.random.normal(size=(n, p))
	groups = np.zeros(n)
	groups[n//2:] = 1
	beta1 = np.random.normal(size=p)
	beta2 = np.random.normal(size=p)
	y = X @ beta1 + (X @ beta2) * groups

	def f(param):
		param1, param2 = param[:p], param[p:]
		preds = X @ param1 + (X @ param2) * groups
		mse = np.mean((y - preds)**2)
		return mse

	x0 = np.random.normal(size=p*2)
	xopt = optimize.minimize(f, x0, method='bfgs', options={'disp': 1})
	assert np.allclose(xopt.x[:p], beta1, atol=1e-4)
	assert np.allclose(xopt.x[p:], beta2, atol=1e-4)
	print(xopt.x)

if __name__ == "__main__":
	test_logistic_model()
	test_contrastive_model()
	test_linear_model()