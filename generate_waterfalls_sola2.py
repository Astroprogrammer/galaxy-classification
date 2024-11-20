
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib as mpl

from matplotlib.collections import LineCollection
import pickle
import glob
import ntpath, os
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap


def plot_rfi(dyn_spectrum,flags,freq_array,time_array,SNR,DM,savefile,PLOTS_DIR):

    fig = plt.figure(figsize=(30,30)); rect = fig.patch
    #fig = plt.figure()
    #fig.suptitle('Ant %d'%(ant))
    #plt.title(label=strategy_name, fontsize=35)
    #plt.title('Ant %d'%(0))
    plt.subplots_adjust(wspace=0.01, hspace=0.26)
    #fig.suptitle(strategy_name+', '+'%.2f'%(flagged_fraction)+'%'+' flagged', fontsize=35,y=0.93,  fontweight  ="bold")
    #fig.suptitle('Waterfall Plot MeerKAT UHF-band (1024 channels)', fontsize=35,y=0.93)
    #fig.suptitle('S/N = %.1f'% (SNR))

    #***************************************************************
    ax1 = fig.add_subplot(311)
    ax1.clear()
    #ax1.title.set_text('Waterfall',size=35)
    ax1.set_title('Data', fontsize=35, color='black')
    #left = frequencies[0]; right = frequencies[-1];  bottom = 0; top = 1;

    #print(left,right,bottom)

    #ff = (np.real(waterfall[0,:,:])+np.real( np.real(waterfall[3,:,:])))/2.
    #ff = (np.real(waterfall[:,:]))
    #vis_flag = vis_flags[0,:,:]

    #vis_flag = np.transpose(flagvalues)

    #print('ff',ff.shape)

    # Creat a mask
    #masked_array = np.ma.array (ff, mask=np.isnan(array))

    #masked_array= np.ma.masked_where( np.transpose(flags) ==1, dyn_spectrum)
    #masked_array= np.ma.masked_where( (flags) ==1, dyn_spectrum)

    #cmap = plt.cm.get_cmap("ocean"); cmap.set_bad('red',1.)

    #masked_array = np.log10( np.abs(masked_array))

    ###ff = masked_array
    #ff =np.log10( np.abs(ff))

    # Boundaries of the waterfall.
    extent_ds = [time_array[0],time_array[-1],freq_array[0],freq_array[-1]]

    print('max dyn_spectrum', np.amax(dyn_spectrum))
    print('extent_ds',extent_ds)

    #plt.imshow(masked_array,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)
    plt.imshow(dyn_spectrum,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)

    plt.xlabel('Time (seconds)',fontsize=35)
    plt.ylabel('Radio frequency (GHz)',fontsize=35)

    #cs = ax1.imshow(ff, interpolation='nearest', aspect=aspect_ratio, vmin=np.percentile(ff,10), vmax=np.percentile(ff,99))
    #cs = ax1.imshow(im, interpolation='nearest', aspect='18.5', vmin=np.percentile(im,10), vmax=np.percentile(im,99))

    #good scale
    #cs = ax1.imshow(ff,cmap=cmap, aspect='auto',alpha=0.9, vmin=np.percentile(masked_array,14), vmax=np.percentile(masked_array,80), extent=(0, ff.shape[1], 0, ff.shape[0]))

    #cs = ax1.imshow(dyn_spectrum, interpolation='nearest',cmap=cmap, aspect='auto',alpha=0.9, vmin=np.percentile(ff,14), vmax=np.percentile(ff,80), extent=(0, ff.shape[1], 0, ff.shape[0]))

    #x_label_list = [str(int(frequencies[0])), str(int((frequencies[-1]-frequencies[0])/4)), str(int((frequencies[-1]+frequencies[0])/2)), str(int(frequencies[-1]))]

    #print('freqs', frequencies[0], frequencies[-1])
    #xx = np.arange(frequencies[0], frequencies[-1], 100, dtype=int)

    #step = (ff.shape[1]-0)/len(xx)

    #yy = np.arange(0, ff.shape[1], np.ceil(step), dtype=int)

    #print('yy',yy,xx)

    #ax1.set_xticks(yy)

    #ax1.set_xticks([0,ff.shape[1]/4,ff.shape[1]/2,ff.shape[1]]) h.ax.tick_params(labelsize=28)

    #ax1.set_xticklabels(xx)

    # cbar = fig.colorbar(cs)
    # ax.set_aspect(2)
    # #ax.set_yscale('log')
    ax1.set_aspect('auto')
    #ax1.set_xlabel(r"Frequency [MHz]", fontsize=35,labelpad=10)
    #ax1.set_ylabel("Time", fontsize=35)

    #ax1.set_xticks(np.arange(0, 1600, 160))
    #ax.set_yticks(np.arange(-.5, 9, 1))
    #ax1.set_xticklabels(np.arange(544, 1090, 55))
    #ax.set_yticklabels(np.arange(0, 10, 1))

    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.tick_params(axis='both', which='minor', labelsize=22)

    #plt.title('S/N = %.1f'% (SNR))
    h = plt.colorbar()
    h.ax.tick_params(labelsize=28)
    h.ax.yaxis.set_ticks_position('left')
    h.set_label('Flux density (arbitrary units)',fontsize=28, labelpad=30)

    # Adding text without box on the plot.
    #ax1.text(ff.shape[1]-step*0.7, 60, 'Mean', color='white', fontsize=40)

    #**************************************************************************
    ax2 = fig.add_subplot(312)
    ax2.clear()
    ax2.set_title('Flags', fontsize=35, color='black')

    #left = frequencies[0]; right = frequencies[-1];  bottom = 0; top = 1;

    #ff = np.real(waterfall[:,:])
    #vis_flag = vis_flags[0,:,:]

    # Creat a mask
    #masked_array= np.ma.masked_where( np.transpose(flags) ==1, ff)

    # Make a copy
    #cmap = plt.cm.get_cmap("ocean")

    # Choose the color
    #cmap.set_bad('red',1.)

    #ff = masked_array
    #ff =np.log10( np.abs(ff))

    #plt.imshow(dyn_spectrum,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)

    #masked_array= np.ma.masked_where( np.transpose(flags) ==1, dyn_spectrum)
    masked_array= np.ma.masked_where( (flags) ==True, dyn_spectrum)
    #masked_array= np.ma.masked_where( (flags) ==99, dyn_spectrum)

    #dyn_spectrum[flags] = 99;
    #dyn_spectrum[~flags] = 0.;
    #masked_array = dyn_spectrum;

    #palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')

    palette = plt.cm.gray.with_extremes(under='black', bad='white')
    cmp=ListedColormap(['black','white'])

    plt.imshow(masked_array,origin='lower',interpolation='None',                cmap=palette,aspect='auto',extent=extent_ds,norm=colors.Normalize(vmin=0, vmax=1.0))

    #ax2[0].imshow(myMatrix.data,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)
    #Default is to apply mask
    #ax2[1].imshow(myMatrix,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)

    #cs = ax2.imshow(ff, interpolation='nearest',cmap=cmap, aspect='auto',alpha=0.9, vmin=np.percentile(ff,14), vmax=np.percentile(ff,80), extent=(0, ff.shape[1], 0, ff.shape[0]))

    #x_label_list = [str(int(frequencies[0])), str(int((frequencies[-1]-frequencies[0])/4)), str(int((frequencies[-1]+frequencies[0])/2)), str(int(frequencies[-1]))]

    #xx = np.arange(frequencies[0], frequencies[-1], 100, dtype=int)

    #step = (ff.shape[1]-0)/len(xx)

    #yy = np.arange(0, ff.shape[1], np.ceil(step), dtype=int)

    #ax2.set_xticks(yy); ax2.set_xticklabels(xx);

    ax2.set_aspect('auto')
    #ax2.set_xlabel(r"Frequency [MHz]", fontsize=35,labelpad=10)
    #ax2.set_ylabel("Time", fontsize=35)

    plt.xlabel('Time (seconds)',fontsize=35)
    plt.ylabel('Radio frequency (GHz)',fontsize=35)

    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='minor', labelsize=22)

    h = plt.colorbar()
    h.ax.tick_params(labelsize=28)
    h.ax.yaxis.set_ticks_position('left')
    #h.ax.yaxis.set_ticks_position('left')
    #h.ax.xaxis.set_label_position('top')
    h.set_label('Flux density (arbitrary units)',fontsize=28, labelpad=30)

    #ax2.text(ff.shape[1]-step*0.7, 60, pols[0], color='white', fontsize=40)

    #**************************************************************************

    #**************************************************************************

    fig.savefig(PLOTS_DIR+savefile+'.png',bbox_inches='tight')
    #fig.savefig(PLOTS_DIR+savefile+'.pdf',bbox_inches='tight')

    plt.close(fig)

#*********************************************************************************

# Sample from a log-uniform distribution.
def loguniform(low_limit,high_limit,n_samples):
 log_low_limit = np.log(low_limit)
 log_high_limit = np.log(high_limit)
 log_samples = np.random.uniform(log_low_limit,log_high_limit,n_samples)
 samples = np.exp(log_samples)
 return samples

# Simulate Gaussian random noise with zero mean and unit variance.
def noise_std_normal(n_freq_bins,n_time_bins):
 noise = np.random.randn(n_freq_bins,n_time_bins)
 return noise

# Simulate a dispersed pulse (Gaussian profile along frequency and time) and superpose it on a noisy background.
def simulate_pulse(f_center,t_center,FWHM_f,FWHM_t,SNR,n_time_bins,n_freq_bins,freq_array,time_array,DM, theta=5):
 # Simulate noisy background.
 noise = noise_std_normal(n_freq_bins,n_time_bins)
 pulse = np.zeros(noise.shape)

 flags = np.zeros(noise.shape,dtype=bool)

 # Convert supplied FWHM widths along frequency and time axes to 1/e widths.
 sigma_f = FWHM_f/np.sqrt(8*np.log(2))
 sigma_t = FWHM_t/np.sqrt(8*np.log(2))

 # Simulate dispersed pulse.
 for i in range(n_freq_bins):
     nu = freq_array[i]
     t_shift = 4.15*DM*(nu**-2. - f_center**-2.)
     #print(t_shift)
     for j in range(n_time_bins):
         t = time_array[j]
         pulse[i,j] = SNR*np.exp(-0.5*((nu - f_center)/sigma_f)**2)*np.exp(-0.5*((t - t_center-t_shift)/sigma_t)**2)

 # Superpose dispersed pulse on top of background.
 signal = noise+pulse

 rms_median = np.median(np.fabs(noise))

 #ind = np.argwhere( signal >  (theta*rms_median) )

 #flags are pixels above certain noise threshold
 flags = (signal>= (theta*rms_median))

 #print('pulse',pulse.shape, 3.*rms_median, flags.shape, pulse, flags, ind)

 return noise,signal, flags

#**********************************************************************************

# Input parameters for the waterfall spectrum
n_time_bins = 1024 # No. of time steps
n_freq_bins = 1024 # No. of channels

time_resol = 10 # Time resolution of data (s)
freq_start = 0.4 # Frequency (GHz) at lower edge of bandpass.
freq_stop = 0.8 # Frequency (GHz) at upper edge of bandpass.

N_RFI = 1 # number of waterfall plots

# Sample signal FWHM from a uniform distribution between pulse_bw_min and pulse_bw_max.
pulse_bw_min = 0.1 # Minimum pulse bandwidth (GHz)
pulse_bw_max = 1.0 # Maximum pulse bandwidth (GHz)

pulse_bw_min = 0.05 # Minimum pulse bandwidth (GHz)
pulse_bw_max = 0.2 # Maximum pulse bandwidth (GHz)

pulse_time_FWHM_min = 1 # Minimum temporal width (s) of a pulse.
pulse_time_FWHM_max = 50 # Largest allowed pulse temporal width (s).

# Range of temporal widths of RFI.
RFI_time_FWHM_min = 1e-2 # Minimum temporal width (s) of an RFI signal.
RFI_time_FWHM_max = 1e-1 # Largest allowed temporal width (s) of an RFI.

RFI_time_FWHM_min = 200 # Minimum temporal width (s) of an RFI signal.
RFI_time_FWHM_max = 300 # Largest allowed temporal width (s) of an RFI.

#RFI_time_FWHM_min = 200 # Minimum temporal width (s) of an RFI signal.
#RFI_time_FWHM_max = 10000*n_time_bins # Largest allowed temporal width (s) of an RFI.

#RFI_freq_FWHM_min = 10 # Minimum temporal width (s) of an RFI signal.
#RFI_freq_FWHM_max = 10000*n_time_bins # Largest allowed temporal width (s) of an RFI.

# SNR of pulses. Sample signal SNRs from a uniform distribution between these limits.
SNR_min = 3.0 # Minimum signal SNR
SNR_max = 15.0 # Maximum signal SNR

#**********************************
OUTPUT_DIR = './data/'
PLOTS_DIR = './plots/'

os.system('mkdir '+PLOTS_DIR)
#os.system('rm -rf ./plots/*')

os.system('mkdir '+OUTPUT_DIR)
#os.system('rm -rf ./plots/*')

##################################################################################

tot_time = time_resol*n_time_bins
bandwidth = freq_start - freq_stop # Total bandwidth (GHz) of data set.
chan_bandwidth = (freq_stop - freq_start)*1e3/n_freq_bins # Channel bandwidth (MHz)
# Array of frequencies corresponding to spectral channels.
freq_array = np.linspace(freq_start,freq_stop,n_freq_bins) # GHz
# Array of time stamps for each pixel of the dynamic spectrum.
time_array = np.linspace(0,tot_time,n_time_bins) # ms


# Center frequencies and times of RFI signals.
f_center_RFI = np.random.uniform(freq_start,freq_stop,N_RFI) # GHz
t_center_RFI = np.random.uniform(0.,tot_time,N_RFI) # ms
# FWHM of RFI signals
FWHM_freq_RFI = np.random.uniform(pulse_bw_min,pulse_bw_max,N_RFI) # GHz
FWHM_time_RFI = loguniform(RFI_time_FWHM_min,RFI_time_FWHM_max,N_RFI)#*1e3 # ms
SNR_RFI = np.random.uniform(SNR_min,SNR_max,N_RFI)

print(FWHM_freq_RFI,FWHM_time_RFI)

do_plot = True
#do_plot = False

#####################################

outfile = OUTPUT_DIR+'test3.pkl'

os.system('rm -rf '+outfile)
open_file = open(outfile, "wb")

#pickle.dump( freq_array, open_file);
#pickle.dump( time_array, open_file);

for i in range(N_RFI):

   print('RFI no.: %d'% (i+1))

   noise, data, flags = simulate_pulse(f_center_RFI[i],t_center_RFI[i],FWHM_freq_RFI[i],FWHM_time_RFI[i],SNR_RFI[i],n_time_bins,n_freq_bins,freq_array,time_array,0)

   savefile = 'a'+str(np.random.randint(100))+'_DM'+str(np.round(0).astype(int))+'_SNR'+str(np.round(SNR_RFI[i],2))

   if do_plot:
      plot_rfi(data,flags,freq_array,time_array,SNR_RFI[i],0.0,savefile,PLOTS_DIR)

   pickle.dump( noise, open_file);
   pickle.dump( data, open_file);
   pickle.dump( flags, open_file);

open_file.close()

#***********************************
