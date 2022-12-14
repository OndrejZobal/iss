import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

import pandas as pd
import seaborn as sns

from ipywidgets import IntSlider

def ipy_random_process(ksi, orig, plot=10, n=50, show_orig=True):
    """ Interactive ... """
    n_plot = plot
    _, ax = plt.subplots(2, 1, sharey='all', figsize=(16, 12))
    ax[0].set_title(f'Zobrazení prvnich {n_plot} realizaci ' +
                    f'nahodneho vyfiltrovaného signalu z {ksi.shape[0]} vygenerovaných')
    ax[0].plot(ksi[:n_plot].T, linewidth=0.7)
    ax[0].set_xlabel('čas $n$')

    
    ax[0].plot([n, n], [ksi.min(), ksi.max()], 'k--', linewidth=2, alpha=0.4)
    ax[0].set_xlim(0-0.2, 200+0.2), 
    ax[0].set_ylim([ksi.min(), ksi.max()])

    if show_orig:
        ax[0].plot(orig[:n_plot].T, '#ff1919', linewidth=0.5, alpha=0.8)

    ax[1].set_title(f'Hodnoty prvnich {n_plot} realizací nahodného signálu v čase $n$ = {n}')
    ax[1].set_xlabel('Číslo realizace')

    markerline, stemlines, baseline = ax[1].stem(ksi[:, n], basefmt="k")
    plt.setp(stemlines, 'color', 'b','linewidth', 2.4)
    plt.setp(markerline, 'color', 'b','markersize', 2)

    if show_orig:
        markerline, stemlines, baseline = ax[1].stem(orig[:, n])
        plt.setp(stemlines, 'color', 'r', 'linewidth', 1.4)
        plt.setp(markerline, 'color', 'r', 'markersize', 2)

    # ax[1].set_xlim(0, omega)  
    if n_plot == 1:
        ax[1].set_xlim(-0.99, 0.99) 
    else:
        ax[1].set_xlim(-0.01*n_plot, 1.01*(n_plot-1)) 
    ax[1].set_ylim(ksi.min(), ksi.max())
    plt.tight_layout()
    
    
def i_CDF_PDF(ksi, n_time=50, n_aprx=40, a=0, b=100, LERP=False):
    omega = ksi.shape[0]
    xmin = np.min(ksi)
    xmax = np.max(ksi)

    n_aprx = n_aprx  # number of values we use to aproximate CDF
    x = np.linspace(xmin, xmax, n_aprx)

    # n_time, for which we estimate CDF
    Fx = np.zeros(x.shape)
    for i in range(n_aprx):
        Fx[i] = np.sum(ksi[:, n_time] < x[i]) / omega
    
    plt.figure(figsize=(16, 6))
    binsize = np.abs(x[1] - x[0])
    hist, _ = np.histogram(ksi[:, n_time], n_aprx)
    px = hist / omega / binsize
    
    if LERP:
        plt.fill_between(x[a:b+1], px[a:b+1], color='#fbeb80', alpha=0.8)
        plt.plot(x, px, color='#f1d518', linewidth=4)
    else:
        plt.fill_between(x[a:b+1], px[a:b+1], step='post', color='#fbeb80', alpha=0.8)  
        plt.plot(x, px, drawstyle='steps-post', color='#f1d518', linewidth=4)

    plt.xlabel('$x$')
    plt.grid(alpha=0.5, linestyle='--')
    plt.xlim(-50, 50) #plt.xlim(xmin, xmax)
    plt.ylim(-0.06/100, 0.06)

    if a == b:
        integral = 0.0 
    else:
        integral = np.sum(px[a:b] * binsize)
    integral_string = '$\int_{a}^{b}$' 
    #plt.title('CDF, PDF ' +integral_string + '$f(x, n$={1}$)$ = {2:2.3}'
        #.format(0, n_time, integral), fontsize=20)
    
    ax = plt.twinx()

    if LERP:
        ax.plot(x[a:b+1], Fx[a:b+1], color='#fd8a21', linewidth=4)
    else:
        ax.plot(x[a:b+1], Fx[a:b+1], color='#fd8a21', drawstyle='steps-post', linewidth=4)

    ax.grid(alpha=0.5, linestyle='--')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()

maxima = 80
iw_a = IntSlider(min=0, max=maxima, step=1)
iw_b = IntSlider(value=80, min=0, max=maxima, step=1)
iw_n_aprx = IntSlider(value=40, min=5, max=maxima, step=5)

def update_a(*args):
    if iw_a.value > iw_b.value: 
        iw_a.value = iw_b.value
def update_b(*args):
    if iw_b.value < iw_a.value: 
        iw_b.value = iw_a.value
def update_upper_limit(*args):
    if iw_n_aprx.value < iw_b.value: 
        iw_b.value = iw_n_aprx.value

iw_b.observe(update_upper_limit, 'value')
iw_b.observe(update_a, 'value')
iw_a.observe(update_b, 'value')


def acorr_coef(ksi, n_bins=40, n1=50, n2=51):
    """ Calculates auto-correlation coefficient. """
    px1x2, x1_edges, x2_edges = np.histogram2d(ksi[:, n1], ksi[:, n2], n_bins, normed=True)
    binsize = np.abs(x1_edges[0] - x1_edges[1]) * np.abs(x2_edges[0] - x2_edges[1])
    
    # x1 * x2
    bin_centers_x1 = x1_edges[:-1] + (x1_edges[1:] - x1_edges[:-1]) / 2
    bin_centers_x2 = x2_edges[:-1] + (x2_edges[1:] - x2_edges[:-1]) / 2
    x1x2 = np.outer(bin_centers_x1, bin_centers_x2)

    # auto-correlation coefficient
    R = np.sum(x1x2 * px1x2 * binsize)
    return R


def ipy_JPDF(ksi, n1=50, n2=51, n_bins=40):
    R = acorr_coef(ksi, n_bins=n_bins, n1=n1, n2=n2)
    data = pd.DataFrame(np.array([ksi[:, n1], ksi[:, n2]]).reshape(2, -1).T)
    data = data.rename(columns={0: 'x1', 1: 'x2'})
    
    with sns.axes_style("darkgrid"):
        g = sns.jointplot(
            data=data, x='x1', y='x2', kind='hist', height=11, 
            cmap=cm.Blues, # color='skyblue',
            cbar=True,
            xlim=(-40, 40), 
            ylim=(-40, 40),
            alpha=0.9,
            marginal_ticks=True, 
            marginal_kws={
                'binwidth': 80/n_bins, 
                'binrange': [-40, 40],
                'color': '#87ceeb',
                'edgecolor': '#063663',
                'linewidth': 1.4,
            },
            joint_kws={'binwidth': 80/n_bins, 'binrange': [-40, 40]},
            zorder=1,
        )
        g.ax_marg_x.yaxis.set_major_locator(MaxNLocator(3))
        g.ax_marg_y.xaxis.set_major_locator(MaxNLocator(3))

        # Texts
        plt.text(x=50, y=50, s='$R(n_1={0}, n_2={1}) = {2:0.2f}$'.format(n1, n2, R))
        kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 24}
        scolor = {'green': '#2b8039', 'red': '#ab1d0a'}
        x = 34
        y = 37
        g.ax_joint.text(x=+x, y=+y, s='$(+,+)$', **kwargs)
        g.ax_joint.text(x=-x, y=-y, s='$(-,-)$', **kwargs)
        g.ax_joint.text(x=+x, y=-y, s='$(+,-)$', **kwargs)
        g.ax_joint.text(x=-x, y=+y, s='$(-,+)$', **kwargs)

        g.refline(x=0, y=0, color='k', linewidth=2, linestyle='-', alpha=0.4, zorder=3)
        
        # get the current positions of the joint ax and the ax for the marginal x
        pos_joint_ax = g.ax_joint.get_position()
        pos_marg_x_ax = g.ax_marg_x.get_position()
        
        # reposition the joint ax so it has the same width as the marginal x ax
        g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, 
                                 pos_marg_x_ax.width, pos_joint_ax.height])

        # reposition the colorbar using new x positions and y positions of the joint ax
        g.fig.axes[-1].set_position([1.02, pos_joint_ax.y0, .07, pos_joint_ax.height])
