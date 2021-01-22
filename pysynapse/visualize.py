import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


def plot_gaussian_mixture(means, variances, proportions):
    n = 10000
    mu = [0, 10]
    sigma = [1, 1]
    samples = []
    samples0 = []
    samples1 = []
    for i in range(n):  # iteratively draw samples
        Z = np.random.choice([0, 1])  # latent variable
        if Z == 0:
            samples0.append(np.random.normal(mu[Z], sigma[Z]))
            samples.append(samples0[-1])
        else:
            samples1.append(np.random.normal(mu[Z], sigma[Z]))
            samples.append(samples1[-1])

    grid = np.linspace(min(samples) - 0.5, max(samples) + 0.5, 1000)
    y = scipy.stats.gaussian_kde(samples).evaluate(grid)
    # Double the number of points to make sure the bandwidth in the KDE will be the same
    y0 = scipy.stats.gaussian_kde(samples0 * 2).evaluate(grid)
    y1 = scipy.stats.gaussian_kde(samples1 * 2).evaluate(grid)
    # Multiply by maximum height to scale
    y /= max(y)
    y0 /= max(y0)
    y1 /= max(y1)
    plt.plot(grid, y0, label='Component 1')
    plt.fill_between(grid, 0, y0, alpha=0.5)
    plt.plot(grid, y1, label='Component 2')
    plt.fill_between(grid, 0, y1, alpha=0.5)
    plt.plot(grid, y, '--', label='Mixture')
    plt.legend()
