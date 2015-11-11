import numpy as np
from sklearn import mixture
#woah sci kit learn is really cool

np.random.seed(1)

mleArr = []
bic = []
mleDifferences = []


def mle(GMM, X):
	return -2 * GMM.score(X).sum()

def verifyConcavity(arr):
	for x in xrange(0, len(arr) - 2):
		if arr[x+2] - 2*arr[x +1] + arr[x] >  0:
			return False
	return True
			

for x in xrange(1,10):
	g = mixture.GMM(n_components=x)
	# generate random observations with two modes centered on 0 and 100 
	obs = np.concatenate((np.random.randn(100, 1), 100 + np.random.randn(300, 1)))
	g.fit(obs)
	print 'mle'
	print mle(g, obs)
	mleArr.append(mle(g, obs))
	bic.append(g.bic(obs))

print mleArr
#why is the fit so good for 1 GMM?
print bic

for x in xrange(1,len(mleArr) - 1):
	mleDifferences.append(mleArr[x] - mleArr[x-1])	

print mleDifferences
print verifyConcavity(mleDifferences)



#bic source code, using this to define MLE function
# def bic(self, X):
# """Bayesian information criterion for the current model fit
# and the proposed data
# Parameters
# ----------
# X : array of shape(n_samples, n_dimensions)
# Returns
# -------
# bic: float (the lower the better)
# """
# return (-2 * self.score(X).sum() +
#         self._n_parameters() * np.log(X.shape[0]))




