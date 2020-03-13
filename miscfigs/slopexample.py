import numpy as np
import matplotlib.pyplot as plt

def linear(x,b,m):
    return b + m*x

def slop_ex():
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    ax = axs

    x = np.linspace(0, 10, 70)
    x_clean = x
    y_clean = linear(x, 1, 0.5)
    y = np.zeros(len(x))
    sx = np.zeros(len(x))
    sy = np.zeros(len(x))

    for i,_ in enumerate(x):
        y[i] = y_clean[i] + np.random.normal(scale=2.0)
        sx[i] = np.abs(np.random.normal(loc=0.3, scale=0.2))
        sy[i] = np.abs(np.random.normal(loc=0.3, scale=0.2))

    
    x = x_clean[20:50]
    y = y[20:50]
    sx = sx[20:50]
    sy = sy[20:50]

    # confidence bands:
    slop = 1

    ax.errorbar(x, y, yerr=sy, xerr=sx,  fmt="o", color="k", ecolor='k', label="Data", markersize=4)
    ax.plot(x_clean, y_clean, label="Model", color="k", linewidth=3, alpha=0.75)
    ax.fill_between(x_clean, y_clean + slop, y_clean - slop, color="k", alpha=0.4)
    ax.fill_between(x_clean, y_clean + 2*slop, y_clean - 2*slop, color="k", alpha=0.3)
    ax.fill_between(x_clean, y_clean + 3*slop, y_clean - 3*slop, color="k", alpha=0.2)

    plt.ylim(1,8)
    plt.xlim(2,8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylabel("$y$")
    # plt.xlabel("$x$")
    # plt.legend()
    plt.show()
    # plt.savefig("slopexample.pdf")

if __name__ == '__main__':
    slop_ex()