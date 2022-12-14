#!/usr/bin/env python3

import numpy as np
import soundfile as sf
from IPython.display import display
from IPython.display import Audio
import matplotlib.pyplot as plt
import scipy

MIDIFROM = 24
MIDITO = 108
SKIP_SEC = 0.25
HOWMUCH_SEC = 0.5
WHOLETONE_SEC = 2

howmanytones = MIDITO - MIDIFROM + 1
tones = np.arange(MIDIFROM, MIDITO+1)
s, Fs = sf.read('../audio/klavir.wav')
N = int(Fs * HOWMUCH_SEC)
Nwholetone = int(Fs * WHOLETONE_SEC)
xall = np.zeros((MIDITO+1, N))


def plot_audio(t, s, fs, ax=None, name='', plot_kwargs={}):
    """  """
    if ax == None:
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(111)
    if name:
        name = ' [' + name + ']'
    ax.plot(t, s, **plot_kwargs)
    ax.set_title('Audio signal'+name, pad=25)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')

    ax.set_xlim(min(t), max(t))
    ylim = (-1., 1.)
    yticks = np.array([-1,-0.5, 0, 0.5, 1])
    for h in [0.5, 0.25, 0.125]:
        if np.max(abs(s)) <= h:
            yticks /= 2.
            ylim = (ylim[0]/2., ylim[1]/2.)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.grid()
    xticks = ax.get_xticks()

    # Second x-axis (samples)
    ax2 = ax.twiny()
    ax2.set_xlabel('Samples', labelpad=10)
    ax2.set_xticks(np.round(xticks*fs).astype(int))
    ax2.set_xlim(min(t)*fs, max(t)*fs)
    plt.tight_layout()

def plot_spectrum(f, g, stop=None, ax=None, plot_kwargs={'color': '#602c69'}, name=""):
    if ax == None:
        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(111)

    if stop == None:
        stop = f.size
    ax.plot(f[:stop], g[:stop], **plot_kwargs)  # <= spectrum
    ax.set_xlim(f[:stop].min(), f[:stop].max())

    ax.set_title(f"Power spectral density (PSD) {name}")
    ax.set_xlabel('Frequency $[Hz]$', labelpad=10)
    ax.set_ylabel('Mag.\n$[dB]$', labelpad=20, rotation=0)

    ax.grid()
    plt.tight_layout()

def post_process_dft_spec(dft_spec):
    G = 10 * np.log10(1/dft_spec.size * np.abs(dft_spec) ** 2)
    freq = np.arange(G.size) * Fs / N
    return (G, freq)

def soundfile_write(filename, s, fs):
    """
    If you're having trouble with playing audio inside notebook change SOUND_WRITE to True,
    then play created files using some program e.g. `Audacity`.
    """
    SOUND_WRITE = False
    if SOUND_WRITE:
        sf.write(filename, s, fs)

## Moje noty
NOTE_1 = 36
NOTE_1_FREQ = 65.41
NOTE_2 = 47
NOTE_2_FREQ = 123.47
NOTE_3 = 87
NOTE_3_FREQ = 1244.51

## 4.1 Základy
samplefrom = int(SKIP_SEC * Fs)
sampleto = samplefrom + N

for tone in tones:
    x = s[samplefrom:sampleto]
    x = x - np.mean(x)
    xall[tone,:] = x
    samplefrom += Nwholetone
    sampleto += Nwholetone

## NOTE 1
display(Audio(xall[NOTE_1], rate=Fs))
lbound = 0
rbound = lbound + int(Fs / NOTE_1_FREQ*3)
sample1 = xall[NOTE_1][lbound:rbound]
# Showing three periods
sample_t = np.arange(sample1.size) / Fs
plot_audio(sample_t, sample1, Fs, name=f"MIDI {NOTE_1} - 3 periods")
# Showing spectrum
tone_t = np.arange(xall[MIDIFROM].size) / Fs
sample1_spec = np.fft.fft(xall[NOTE_1])
sample1_g, sample1_freq = post_process_dft_spec(sample1_spec)
#plot_spectrum(sample1_freq, sample1_g, stop=sample1_freq.size//2, name=f"MIDI {NOTE_1}")

## NOTE 2
display(Audio(xall[NOTE_2], rate=Fs))
lbound = 250
rbound = lbound + int(Fs / NOTE_2_FREQ*3)
sample2 = xall[NOTE_2][lbound:rbound]
# Showing three periods
sample_t = np.arange(sample2.size) / Fs
plot_audio(sample_t, sample2, Fs, name=f"MIDI {NOTE_2} - 3 periods")
# Showing spectrum
tone_t = np.arange(xall[MIDIFROM].size) / Fs
sample2_spec = np.fft.fft(xall[NOTE_2])
sample2_g, sample2_freq = post_process_dft_spec(sample2_spec)
#plot_spectrum(sample2_freq, sample2_g, stop=sample2_freq.size//2, name=f"MIDI {NOTE_2}")

## NOTE 3
display(Audio(xall[NOTE_3], rate=Fs))
lbound = 0
rbound = lbound + int(Fs / NOTE_3_FREQ*3)
sample3 = xall[NOTE_3][lbound:rbound]
# Showing three periods
sample_t = np.arange(sample3.size) / Fs
plot_audio(sample_t, sample3, Fs, name=f"MIDI {NOTE_3} - 3 periods")
# Showing spectrum
tone_t = np.arange(xall[MIDIFROM].size) / Fs
sample3_spec = np.fft.fft(xall[NOTE_3])
sample3_g, sample3_freq = post_process_dft_spec(sample3_spec)
#plot_spectrum(sample3_freq, sample3_g, stop=sample3_freq.size//2, name=f"MIDI {NOTE_3}")

## 4.2 Určení základní frekvence

def max_index(array):
    max_value = array[0]
    max_index = 0

    for i, val in enumerate(array):
        if max_value <= val:
            val = max_value
            max_index = i
    return max_index

def compute_base_freq_dft(array):
    spectrum = np.fft.fft(array)
    maximal = np.argmax(abs(spectrum))
    return maximal * (Fs / array.size)

def autocorr(x):
    result = np.correlate(x, x, mode='same')
    return result[result.size//2:]

def compute_base_freq_autoco(array):
    ac = autocorr(array)
    peaks = scipy.signal.find_peaks(ac, height=max(ac))[0]
    period = abs(peaks[0]-peaks[1])
    return 1/period*Fs

for note_index in range(MIDIFROM, MIDITO):
    if note_index <= 56:
        base = compute_base_freq_dft(xall[note_index])
    else:
        base = compute_base_freq_autoco(xall[note_index])
    print(f"MIDI note {note_index} has a base frequency of {base}")

"""
def autocorr(x):
    result = np.correlate(x, x, mode='same')
    return result[result.size//2:]

def max_index(array):
    array = abs(array)
    max_val = array[0]
    max_i = 0

    for index, val in enumerate(array):
        if max_val <= val:
            max_val = val
            max_i = index
    return max_i

def calc_base_freq_autocor(array):
    ac = autocorr(array)
    peaks = scipy.signal.find_peaks(ac, height=max(ac)/1.5)[0]
    period = abs(peaks[0]-peaks[1])
    return 1/period*Fs

def calc_base_freq_fft(array):
    spectrum = np.fft.fft(array)
    return max_index(spectrum[spectrum.size//2:]) * Fs / array.size

for i in range(MIDIFROM, MIDITO):
    if i <= 56:
        base_freq = calc_base_freq_autocor(xall[i])
    else:
        base_freq = calc_base_freq_fft(xall[i])
    print(f"MIDI #{i} has a base frequency of {base_freq}Hz")
    if i in [NOTE_1, NOTE_2, NOTE_3]:
        pass

base_freq1 = calc_base_freq(sample1)
print("Základní frekvence 1. vzorku:", base_freq1)

base_freq2 = calc_base_freq(sample2)
print("papada", base_freq2)
print("Základní frekvence 2. vzorku:", base_freq2)

base_freq3 = calc_base_freq(sample3)
print("Základní frekvence 3. vzorku:", base_freq3)
#plot_spectrum(sample1_freq, abs(sample1_spec), stop=700, name=f"MIDI {NOTE_1}")
#plot_spectrum(sample2_freq, abs(sample2_spec), stop=700, name=f"MIDI {NOTE_2}")
#plot_spectrum(sample3_freq, abs(sample3_spec), stop=700, name=f"MIDI {NOTE_3}")

"""
plt.show()
