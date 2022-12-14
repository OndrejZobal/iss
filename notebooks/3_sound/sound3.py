import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
import os

fontsize=16
fc_color='#fff'

def plot_audio2(t, s, ax=plt, color='#254e57', linewidth=0.2, alpha=0.75):
    #ax.figure(figsize=(18, 4), facecolor=fc_color)
    ax.plot(t, s, color=color, linewidth=linewidth, alpha=alpha)

    t_low = min(t)
    t_upp = max(t)    
    #ax.title("Audio segment ({0:2.2f} to {0:2.2f} s).".format(t_low, t_upp),
    # fontsize=fontsize)
    ax.axis([t_low, t_upp, -1, 1])

    # X
    ax.set_xlabel('Time $[s]$', fontsize=fontsize, labelpad=10)
    # ax.set_xticks(fontsize=fontsize-2)  # ax.xticks(np.arange(round(t_upp)+1))

    # Y
    ax.set_ylabel('Signal magnitude', fontsize=fontsize, labelpad=20)
    ylim = (-1., 1.)
    yticks = np.array([-1,-0.5, 0, 0.5, 1])
    for h in [0.5, 0.25, 0.125]:
        if np.all(abs(s) <= h):
            yticks /= 2.
            ylim = (ylim[0]/2., ylim[1]/2.) 
    ax.set_ylim(ylim)
    ax.set_yticks(yticks, fontsize=fontsize-2)

    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(True)

    
def plot_audio(x, y, ax):
    t_low = min(x)
    t_upp = max(x)    
    ax.axis([t_low, t_upp, -1, 1])

    # X
    ax.set_xlabel('time $[s]$', fontsize=fontsize, labelpad=10)
    # ax.set_xticks(fontsize=fontsize-2)  # ax.xticks(np.arange(round(t_upp)+1))

    # Y
    ax.set_ylabel('magnitude', fontsize=fontsize, labelpad=20)
    ylim = (-1., 1.)
    yticks = np.array([-1,-0.5, 0, 0.5, 1])
    for h in [0.5, 0.25, 0.125]:
        if np.all(abs(y) <= h):
            yticks /= 2.
            ylim = (ylim[0]/2., ylim[1]/2.) 
    ax.set_ylim(ylim)
    ax.set_yticks(yticks, fontsize=fontsize-2)

    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(True)
 

def plot_spectrum(f, s, linewidth=1, color="#602c69"):
    plt.figure(figsize=(18, 6), facecolor=fc_color)
    plt.plot(f, s, color=color, linewidth=linewidth)
    
    f_LowerLimit = min(f)
    f_UpperLimit = max(f)
    plt.title("Power spectral density", fontsize=fontsize)
    plt.axis([f_LowerLimit, f_UpperLimit, -80, 0])

    # X
    plt.xlabel('Frequency $[Hz]$', fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=fontsize-2)

    # Y
    plt.ylabel('Magnitude $[dB]$', fontsize=fontsize, labelpad=20)
    plt.yticks(fontsize=fontsize-2)  #yticks = [-1,-0.5, 0, 0.5, 1]
    
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_spectrogram(time, freq, sgr_log):
    plt.figure(figsize=(20, 6))
    plt.title("Spectogram", fontsize=fontsize, pad=10)
    plt.pcolormesh(time, freq, sgr_log, shading="gouraud", cmap=cm.inferno, vmin=-160)

    # X
    plt.gca().set_xlabel('Time $[s]$', fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2, ticks=np.arange(round(max(time), 1)+1))
    plt.xlim(round(min(time), 1), round(max(time), 1))
    
    # Y
    plt.gca().set_ylabel('Frequency\n$[Hz]$', 
        rotation=0, fontsize=fontsize-2, labelpad=42)
    plt.yticks(fontsize=fontsize-2)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('PSD\n$[dB]$', rotation=0, labelpad=30, fontsize=fontsize-2)

    plt.tight_layout()
    plt.show()


# Save figure
def _save_plot(loc):
    if loc is not None:
        if '/' in loc:
            os.makedirs(os.path.dirname(loc), exist_ok=True)
        plt.savefig(loc, bbox_inches='tight')