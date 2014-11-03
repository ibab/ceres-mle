
from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

def nbins(x):
    n =  (max(x) - min(x)) / (2 * len(x)**(-1/3) * (percentile(x, 75) - percentile(x, 25)))
    return min(n, 150)

data = genfromtxt('data.txt')
params = genfromtxt('params.txt')

h, bins = histogram(data, bins=nbins(data))
width = bins[1] - bins[0]
center = (bins[:-1] + bins[1:]) / 2
xs = linspace(-4, 4, 2000)
y = sum(h * width) * norm.pdf(xs, *params)

plt.errorbar(center, h, sqrt(h), fmt='ko', markersize=5)
plt.plot(xs, y, 'b-')
plt.xlim(-5, 5)
plt.savefig('out.pdf')

plt.clf()
sm.qqplot(data, line='45')
plt.savefig('qq.pdf')

