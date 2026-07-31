from __future__ import division,print_function
import os,sys
import pylab as ppl
import numpy as np
from numpy import pi, e, cos, sin
from os import path,walk
import matplotlib.ticker as ticker
from scipy.signal import argrelmax
from scipy.interpolate import UnivariateSpline as spline

from scipy.stats import scoreatpercentile as scp
from skimage import io
from scipy import ndimage
from numpy.ma import masked_array
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# our libraries
import wavelet_analysis as wa
import kymo_lib as kl

#%%
ppl.ion()

#=======Defaults=============================================
#dt = 0.05*60
dt = 10 # sampling interval

# cut-off period for the sinc filter
#T_c =250 #150
T_c = 200 #200; 6h cycles
#periods = np.linspace(70,T_c,180) # for the wavelet transform, narrow freq. band recommended! % [50, 200]

periods = np.linspace(50,T_c,150)  #[50,Tc,150]: 6h cycles
#[10 200] for regular intensity kymos 

wAn = wa.TFAnalyser(periods,dt,T_c) 

resdir = './res/'
#datadir = '/Users/moenke/Desktop/Rectangles/'
#datadir = path.expanduser('~/PSM/data/TakehitoKymos/') # location of kymos

savedir = '/Volumes/sapna4tb/20200304_medaka/tailImaging/chronic_27C/pooledAnalyses/analysesWtihoutRegistration2/Kymo/';
datadir = os.path.join(savedir, 'intensityKymo/')

# find all tifs in datadir
fnames = list(walk(datadir))[0][2]

fnames = [n for n in fnames if 'tif' in n]
#fnames = [n for n in fnames if 'Resli' in n]

# non-isotropic kymos smoothing
sigma_x = 5# 5, 20(16bit)
sigma_t = 1 # no time smoothing

# ik1 = read_image(fnames[0],datadir)
##%%
def read_all():
    imgs = {}
    for name in fnames:
        if name.startswith('2'): #sapna
            ik = kl.read_image(name,datadir)
            imgs[name[:-4]] = ik
    return imgs

imgs = read_all()
names = [key for key in imgs]
# select by name
name = names[0]

def analyse(name):
    print('working on {}'.format(name))

    #----truncate kymo-------------------------
    chop = 0 # cut off at the beginning
    ik = imgs[name][:,:] # the raw kymo
    #------------------------------------------

    # equal aspect
    aspect = ik.shape[1]/ik.shape[0]

    # detrending and blurring
    dik = kl.detrend_Kymo(ik,dt,T_c = T_c)
    dgik = kl.gblur(dik,(sigma_x,sigma_t))
    #dgik = dik

    # masking/segmentation of the oscillatory field
    mik = kl.mk_intKymo_mask(ik,min_int = ppl.median(ik),Plot = True,win_len = 37,two_sided= True)
    ax = ppl.gca() ; ax.set_title(name)
#%%
    # calculating the phase kymos
    hk = kl.hilbert_Kymo(dgik) # complex signal from hilbert
    wk,twk,pk = kl.wavelet_Kymo(dgik,wAn) # complex wavelets and period kymos, added pwk(wavelet power) - sapna
#%%

    hpk = np.angle(hk) # hilbert phase kymo
    wpk = np.angle(wk) # wavelet phase kymo
    wak = np.abs(wk)   # wavelet amplitude kymo

    # masking the phase kymos
    #mphk = masked_array(hpk,mask = mik.mask)
    #mwpk = masked_array(wpk,mask = mik.mask)

    #kl.Plot_Kymo(mphk,cmap = 'jet',vmin = - pi,vmax = pi, num = 2,offset = chop*dt)
    #ax = ppl.gca() ; ax.set_title(name)
    #ax.imshow(hpk,alpha = 0.25,cmap = 'jet', aspect = aspect)

    #kl.Plot_Kymo(mwpk,cmap = 'jet',vmin = - pi,vmax = pi, num = 12,offset = chop*dt)
    #ax = ppl.gca() ; ax.set_title(name)
    #ax.imshow(wpk,alpha = 0.25,cmap = 'jet',aspect = aspect)

    # amplitude kymo
    #kl.Plot_Kymo(wak,cmap = 'afmhot', num = 13,offset = chop*dt)

    # continue analysis of the kymographs

    # time points in indices
    #epochs = [40,60,80,100]
    # phase profiles
    #kl.Plot_phase_profiles(mwpk, epochs = epochs, dt = dt)

    # plot amplitude profiles
    #kl.Plot_ampl_profiles(wak, epochs = epochs, dt = dt)

    # extract the wavenumber
    #kl.extract_q(mwpk, dt = 10, label = True)

    
    out_path = os.path.join(savedir, 'phasekymo/phase_' + name + '.tif')
    io.imsave(out_path, np.float32(wpk))
    print('written',out_path)

    out_path = os.path.join(savedir, 'phasekymo/complex_phase_' + name + '.npy')
    np.save(out_path, wk)
    print('written',out_path)

    out_path = os.path.join(savedir, 'periodkymo/period_' + name + '.tif')
    io.imsave(out_path, np.float32(twk))
    print('written',out_path)

    out_path = os.path.join(savedir, 'amplitudekymo/amplitude_' + name + '.tif')
    io.imsave(out_path, np.float32(wak))
    print('written',out_path)

    out_path = os.path.join(savedir, 'detrendedkymo/detrended_' + name + '.tif')
    io.imsave(out_path, np.float32(dik))
    print('written',out_path)
    
    out_path = os.path.join(savedir, 'detrendedBlurkymo/detrendedBlur_' + name + '.tif')
    io.imsave(out_path, np.float32(dgik))
    print('written',out_path)
    
    out_path = os.path.join(savedir, 'waveletPowerkymo/waveletPower_' + name + '.tif')
    io.imsave(out_path, np.float32(pk))
    print('written',out_path)
    
    
    #print(ik.shape)
    #return mik

if not os.path.isdir(savedir + '/phasekymo'):
    os.mkdir(savedir + '/phasekymo')
if not os.path.isdir(savedir + '/periodkymo'):
    os.mkdir(savedir + '/periodkymo')
if not os.path.isdir(savedir + '/amplitudekymo'):
    os.mkdir(savedir + '/amplitudekymo')
if not os.path.isdir(savedir + '/detrendedkymo'):
    os.mkdir(savedir + '/detrendedkymo')
if not os.path.isdir(savedir + '/detrendedBlurkymo'):
    os.mkdir(savedir + '/detrendedBlurkymo')
if not os.path.isdir(savedir + '/waveletPowerkymo'):
    os.mkdir(savedir + '/waveletPowerkymo')    
    
#%%

for i in names:
    analyse(i)

    
