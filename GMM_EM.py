import numpy as np
import seaborn as sns
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg

color_iter = itertools.cycle(['navy', 'gold', 'c'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(2., 4.)
    plt.ylim(0., 3.)
    #plt.xticks(())
    #plt.yticks(())
    plt.title(title)
    plt.xlabel('Sepal Width in cm')
    plt.ylabel('Petal Width in cm')
    plt.savefig('Sepal Width vs Petal Width.png')

#plot_results(data[:,:2], labels, mean_arr[:,:2], sigma_arr[:,:2,:2], 0, 'Gaussian Mixture')
#plot_results(data[:,1:3], labels, mean_arr[:,1:3], sigma_arr[:,1:3,1:3], 0, 'Gaussian Mixture')
#plot_results(data[:,2:4], labels, mean_arr[:,2:4], sigma_arr[:,2:4,2:4], 0, 'Gaussian Mixture')
#newdata=np.concatenate((data[:,1:2],data[:,3:4]),axis=1)
#newmean=np.concatenate((mean_arr[:,1:2],mean_arr[:,3:4]),axis=1)
#newsigma1=np.concatenate((sigma_arr[:,1:2,1:2],sigma_arr[:,1:2,3:4]),axis=2)
#newsigma2=np.concatenate((sigma_arr[:,3:4,1:2],sigma_arr[:,3:4,3:4]),axis=2)
#newsigma=np.concatenate((newsigma1,newsigma2),axis=1)
#plot_results(newdata, labels, newmean, newsigma, 0, 'Gaussian Mixture')


def loglikelihood():
    logl=0
    for i in range(0,m):
        temp=0
        for j in range(0,k):
            temp=temp+ sp.multivariate_normal.pdf(data[i,:],mean_arr[j,:],sigma_arr[j,:])*phi[j]
        logl=logl+np.log(temp)
    return logl

def e_step():
    #Finding Membership Weights ie matrix Z
    for i in range(0,m):
        den =0
        for j in range(0,k):
            num = sp.multivariate_normal.pdf(data[i,:],mean_arr[j,:],sigma_arr[j,:])*phi[j]
            den=den+num
            Z[i,j]=num
        Z[i,:]=Z[i,:]/den
        assert Z[i, :].sum() - 1 < 1e-4  # Program stop if this condition is false

def m_step():
    for j in range(0,k):
        const=Z[:,j].sum()
        phi[j]=const/m
        
        mu_j=np.zeros(n)
        sigma_j=np.zeros((n,n))
        
        for i in range(0,m):
            mu_j =mu_j + (data[i, :]*Z[i, j])
            #sigma_j =sigma_j+ Z[i, j] * ((data[i, :] - mean_arr[j, :]) * (data[i, :] - mean_arr[j, :]).T) #CHECK .T
        mean_arr[j] = mu_j / const
        
        for i in range(0,m):
            sigma_j =sigma_j+ Z[i, j] * ((data[i:i+1, :] - mean_arr[j:j+1, :]).T * (data[i:i+1, :] - mean_arr[j:j+1, :])) #CHECK .T
        sigma_arr[j] = sigma_j/const+np.eye(n)*0.001
        

iris = sns.load_dataset("iris")
data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris['species'].values


k=3
m=data.shape[0]
n=data.shape[1]

mean_arr = np.random.random((k,n))+np.mean(data)
#covar=np.cov(data.T,bias=True)
sigma_arr = np.array([np.identity(n)+np.random.random((n,n)) for i in range(k)])
phi = np.ones(k)/k             #ALPHAS
Z = np.empty((m,k), dtype=float)   #MEMBERSHIP WEIGHTS
tol=1e-4

#FITTING NOW
logl=1
previous_logl = 0
num_iters = 0
while(logl-previous_logl > tol):
    previous_logl =loglikelihood()
    e_step()
    m_step()
    num_iters =num_iters+ 1
    logl = loglikelihood()
    print('Iteration %d: log-likelihood is %.6f'%(num_iters, logl))
print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, logl))

labels=np.zeros((150,))
for i in range(0,150):
    labels[i]=np.argmax(Z[i,:])

