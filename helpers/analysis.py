import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.special import softmax
import sklearn.datasets
from sklearn.decomposition import FastICA, PCA

def ex11_generate_data():
    mean = [0, 10]
    cov = [[10, 0], [0, 1]]
    X = np.random.multivariate_normal(mean, cov, 1000)
    return X @ np.array(((np.cos(np.pi/6), -np.sin(np.pi/6)), (np.sin(np.pi/6), np.cos(np.pi/6))))

def ex13_generate_data():
    return sklearn.datasets.fetch_olivetti_faces(data_home="data", shuffle=False, random_state=0, download_if_missing=True)["data"]

def ex21_generate_data():
    mean = [0, 10]
    cov = [[10, 0], [0, 0.2]]
    X1 = np.random.multivariate_normal([0,0], cov, 1000)
    X2 = np.random.multivariate_normal([0,0], cov, 1000)
    X = np.concatenate((
        X1 @ np.array(((np.cos(np.pi/6), -np.sin(np.pi/6)), (np.sin(np.pi/6), np.cos(np.pi/6)))) + mean,
        X2 @ np.array(((np.cos(np.pi/3), -np.sin(np.pi/3)), (np.sin(np.pi/3), np.cos(np.pi/3)))) + mean,
    ))
    return X

def ex2_plot_data(data):
    fig, ax = plt.subplots(nrows=len(data.columns)-1, figsize=(20,10), sharex=True)
    for ind, column in enumerate(data.columns[1:]):
        ax[ind].plot(data["seconds_elapsed"], data[column], label=column)
        ax[ind].set_xlim(5, data["seconds_elapsed"].iloc[-1])
        ax[ind].legend(loc="upper right")
        ax[ind].grid(True)
    ax[-1].set_xlabel("time elapsed (sec)")

def ex3_generate_data():

    # generate data
    X1 = np.random.multivariate_normal([1,2], [[0.5,0],[0,0.5]], size=(200,))
    X2 = np.random.multivariate_normal([-2,4], [[0.5,0],[0,0.5]], size=(200,))
    X3 = np.random.multivariate_normal([0,-1], [[0.5,0],[0,0.5]], size=(200,))

    # append and shuffle data
    X = np.concatenate((X1,X2,X3))
    np.random.shuffle(X)

    # return matrices
    return X

def ex36_generate_data():

    # generate data
    X1 = np.random.multivariate_normal([0,1], [[2,0],[0,0.01]], size=(500,))
    X2 = np.random.multivariate_normal([1,0], [[0.01,0],[0,2]], size=(500,))

    # append and shuffle data
    X = np.concatenate((X1,X2))
    np.random.shuffle(X)

    # rotate data slightly
    X = X @ np.array(((np.cos(np.pi/6), -np.sin(np.pi/6)), (np.sin(np.pi/6), np.cos(np.pi/6))))

    # return matrices
    return X

def ex4_plot_GMM_1D():

    # generate data
    X1 = np.random.normal(0, 3, size=(7000,))
    X2 = np.random.normal(3, 0.5, size=(3000,))

    # append and shuffle data
    X = np.concatenate((X1,X2))
    np.random.shuffle(X)

    # plot stuff
    x = np.arange(-10, 10, 0.01)
    pdf1 = 0.7/np.sqrt(2*np.pi*3**2)*np.exp(-1/2/3/3*np.abs(x)**2)
    pdf2 = 0.3/np.sqrt(2*np.pi*0.5**2)*np.exp(-1/2/0.5/0.5*np.abs(x-3)**2)
    plt.figure()
    plt.hist(X, bins=50, density=True)
    plt.plot(x, pdf1, color="blue")
    plt.plot(x, pdf2, color="blue")
    plt.plot(x, pdf1 + pdf2, color="red")
    plt.grid()
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")

    # return 
    return

def ex4_plot_GMM(X, means, covs, rho):

    # create figure 
    fig = plt.figure(figsize=(20,5))

    ##############
    # First plot #
    ##############

    # create axis 
    ax = fig.add_subplot(1, 3, 1)

    # plot data
    ax.scatter(X[:,0], X[:,1], 5)
    ax.scatter(means[:,0], means[:,1], 75, color="red", marker="x")

    # plot ellipse
    for k in range(len(rho)):

        # calculate eigenvalues of covariance matrix
        lambda_, v = np.linalg.eig(covs[k,:,:])
        lambda_ = np.sqrt(lambda_)

        # create ellipse and add to plot
        ell = Ellipse(xy=(means[k,0], means[k,1]),
                      width=lambda_[0]*2, height=lambda_[1]*2,
                      angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor("red")
        ax.add_artist(ell)

    # change appearance
    ax.grid(), ax.set_aspect("equal")
    ax.set_title("data visualization")

    ###############
    # Second plot #
    ###############

    # create axis
    ax = fig.add_subplot(1, 3, 2, projection='3d')

    # generate data for gaussian surf plot
    xx, yy = np.meshgrid(np.arange(-5,5,0.01), np.arange(-5,5,0.01))
    coordinates = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.zeros(np.shape(xx))
    for k in range(len(rho)):
        zz += np.reshape(multivariate_normal.pdf(coordinates.T, mean=means[k,:], cov=covs[k,:,:]), np.shape(zz))

    # create surf plot
    ax.plot_surface(xx, yy, zz, cmap="jet")
    
    # change appearance
    ax.view_init(45, 210) 
    ax.set_zlabel("$p(x)$")   
    ax.set_title("probability density function")

    ##############
    # Third plot #
    ##############

    # create axis
    ax = fig.add_subplot(1, 3, 3)

    # print warning if the number of clusters is not equal to 2
    if len(rho) != 2:
        print("The third plot only supports 2 clusters. The plotted decision boundary is incorrect.")

    # calculate posterior class probability
    r1 = np.log(rho[0]) + np.reshape(multivariate_normal.logpdf(coordinates.T, mean=means[0,:], cov=covs[0,:,:]), np.shape(zz))
    r2 = np.log(rho[1]) + np.reshape(multivariate_normal.logpdf(coordinates.T, mean=means[1,:], cov=covs[1,:,:]), np.shape(zz))
    rhox = softmax(np.stack([r1, r2], axis=2), axis=2)[:,:,0]
    
    # plot map
    ax.imshow(rhox, origin="lower", extent=[-5,5,-5,5])
    ax.scatter(X[:,0], X[:,1], 5)

    # change appearance
    ax.set_title("decision boundary")

    # return 
    return

def plot_gmm(ax, gmm):

    means = gmm.means_
    covs = gmm.covariances_
    rho = gmm.weights_

    # plot means
    ax.scatter(means[:,0], means[:,1], 75, color="red", marker="x")

    # plot ellipse
    for k in range(len(rho)):

        # calculate eigenvalues of covariance matrix
        if len(np.shape(covs)) == 2:
            lambda_, v = np.linalg.eig(np.diag(covs[k,:]))
        else:
            lambda_, v = np.linalg.eig(covs[k,:,:])
        lambda_ = np.sqrt(lambda_)

        # create ellipse and add to plot
        ell = Ellipse(xy=(means[k,0], means[k,1]),
                      width=lambda_[0]*2, height=lambda_[1]*2,
                      angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor("red")
        ax.add_artist(ell)


def plot_eigen(mean, eigvals, eigvecs, ax, **kwargs):
    ax.arrow(mean[0], mean[1], np.sqrt(eigvals[0])*eigvecs[0,0], np.sqrt(eigvals[0])*eigvecs[1,0], **kwargs)
    ax.arrow(mean[0], mean[1], np.sqrt(eigvals[1])*eigvecs[0,1], np.sqrt(eigvals[1])*eigvecs[1,1], **kwargs)

def plot_faces(X):
    fig, ax = plt.subplots(figsize=(10,10), ncols=10, nrows=10)
    for k in range(100):
        ax[k//10, k%10].imshow(np.abs(X[k]).reshape(64,64), cmap="gray")
        ax[k//10, k%10].axis("off")

def plot_pca(ax, pca, mean):
    x_axis, y_axis = pca.components_.T
    ax.quiver(
        (mean[0], mean[0]),
        (mean[1], mean[1]),
        x_axis,
        y_axis,
        zorder=11,
        width=0.01,
        scale=6,
        color="red",
        label="PCA",
    )
def plot_ica(ax, ica, mean):
    x_axis, y_axis = ica.mixing_
    ax.quiver(
        (mean[0], mean[0]),
        (mean[1], mean[1]),
        x_axis/75,
        y_axis/75,
        zorder=11,
        width=0.01,
        scale=6,
        color="orange",
        label="ICA",
    )