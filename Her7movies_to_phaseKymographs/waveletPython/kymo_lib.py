from __future__ import division,print_function
import os,sys
import pylab as ppl
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

sys.path.append(path.expanduser('~/PSM/progs/lib/'))
#import morphsnakes
from wavelet_analysis import *

# rc('lines', markeredgewidth = 0)
# rc('lines', linewidth = 2)

def wavelet_Kymo(dKymo,wAn):
    
    
    wKymo = np.zeros( dKymo.shape, dtype = complex )
    tKymo = np.zeros( dKymo.shape, dtype = float )
    powerKymo = np.zeros( dKymo.shape, dtype = float)

    for i,dtraj in enumerate(dKymo):
        wAn.compute_spectrum(dtraj,Plot = False)
        ridge_data = wAn.get_maxRidge()
        wKymo[i,:] = ridge_data['z']
        tKymo[i,:] = ridge_data['periods'] # the periods
        powerKymo[i,:] = ridge_data['power'] # wavelet power -sapna
        
    return wKymo, tKymo, powerKymo

def wavelet_pKymo(dKymo,wAn):
    
    
    pKymo = np.zeros( dKymo.shape, dtype = float )


    for i,dtraj in enumerate(dKymo):
        wAn.compute_spectrum(dtraj,Plot = False)
        ridge_data = wAn.get_maxRidge()
        pKymo[i,:] = angle(ridge_data['z'])

    return pKymo


def detrend_Kymo(Kymo, dt,T_c = 400):
    
    Nt = Kymo.shape[-1]
    dKymo = np.zeros( Kymo.shape )
    for i,traj in enumerate(Kymo):

        dtraj = sinc_detrend(traj,dt = dt,T_c = T_c)
        dKymo[i,:] = dtraj

    return dKymo

def detrend_mKymo(mKymo, dt,T_c = 400):

    '''
    Detrend masked Kymographs.
    '''

    if type(mKymo) is not masked_array:
        print ('Mask the Kymo first!')
        #return
    Nt = mKymo.shape[-1]
    dKymo = np.zeros( mKymo.shape )
    for i,traj in enumerate(mKymo):

        vb = ~traj.mask
        vtraj = traj[vb] # get only valid points

        if np.size(vtraj) > 4:
            dtraj = sinc_detrend(vtraj,dt = dt,T_c = T_c)
            dKymo[i,vb] = dtraj
            
    return dKymo

def hilbert_pKymo(dKymo):

    ''' detrend beforehand! '''
    Nt = dKymo.shape[-1]
    hKymo = np.zeros( dKymo.shape )
    for i,dtraj in enumerate(dKymo):

        htrafo = hilbert(dtraj)
        phis = angle(htrafo) # extracting the phase from the complex analytic signal

        hKymo[i,:] = phis

    return hKymo


def hilbert_Kymo(dKymo):

    ''' detrend beforehand! '''
    
    Nt = dKymo.shape[-1]
    hKymo = np.zeros( dKymo.shape,dtype = complex )
    for i,dtraj in enumerate(dKymo):

        htrafo = hilbert(dtraj)
        #phis = angle(htrafo) # extracting the phase from the complex analytic signal

        hKymo[i,:] = htrafo

    return hKymo


def get_periods(traj, tfa, detrend = False):

    tfa.compute_spectrum(dtraj, Plot = False, detrend = True)
    ridge_dic = tfa.get_maxRidge()

    periods = ridge_dic['ridge']

    
    return periods

def prune_pprofile(raw_profile, max_deriv = 0.5, Plot = False):
    
    ''' 
    Cut off at potential discontinous jumps of the phase values (after unwrapping) 
    along the symmetric two-sided phase profile. 

    Expects a masked phase array as input. 
    Cut-off value for 1st derivative dependent on pre-processing steps (smoothing etc.).

    Heuristically expects the cut-off values on both the far ends of the profile.

    Plotting only for isolated inspection calls!

    Returns the unwrapped, pruned phase profile and the offset to keep position.
    '''

    # default is no pruning
    right = -1
    left = 0

    valid_inds = np.where(~raw_profile.mask)[0]
    if not np.size(valid_inds):
        raise ValueError('input profile has no valid points!')

    ind_offset = valid_inds[0] # first index with a valid (unmasked) value
    
    vb = ~raw_profile.mask
    profile = np.unwrap(raw_profile[vb])

    a1 = np.arange(len(profile))

    midpoint = int( np.rint(len(profile)/2.) ) # force an integer
    
    ev1 = np.linspace(0, a1[-1], 5000) # evaluation vector just for inspection

    spl = spline(a1,profile, s = 0)
    dspl = spl.derivative()
    
    # the 1st derivative along the unwrapped profile at the data points
    deriv = dspl(a1) 

    jumps = np.where(abs(deriv) > max_deriv)[0]

    # left pruning, most inner point
    lpoints = jumps[jumps < midpoint]
    if np.size(lpoints):
        left = max(lpoints)
        left = left

    # right pruning, most inner point
    rpoints = jumps[jumps > midpoint]
    if np.size(rpoints):
        right = min(rpoints)
        right = right

    # for inspection/debugging only
    if Plot:

        ppl.figure(9);ppl.clf()
        plot(ev1,dspl(ev1))
        plot(a1,dspl(a1),'o')
        plot(a1[left], dspl(a1[left]), 'ko')
        plot(a1[right], dspl(a1[right]), 'ko')

        ppl.figure(10);ppl.clf()
        plot(profile,'o')
        plot(ev1,spl(ev1))

        plot(a1[left], profile[left], 'ko')
        

        plot(a1[right], profile[right], 'ko')

    #print (left,right)
    return profile[ left : right ], ind_offset+left


def extract_q(pKymo, dt, label = None,offset = 0, max_prune_deriv = .5,Plot = True, pruning = True):

    if type(pKymo) is not masked_array:
        print ('constructing empty mask..')
        pKymo = masked_array(data = pKymo, mask = np.zeros(pKymo.shape,dtype = bool))

    
    tvec = np.arange(0,pKymo.shape[1]*dt,dt) + offset*dt
    
    phases_d = []
    phases_up = []

    q_d = []
    q_up = []

    profiles = []
    ts = [] # time points for valid q-extraction
    
    for j,col in enumerate(pKymo.T):

        vb = ~col.mask
        profile = np.unwrap(col[vb]) # get only valid points for the detrending
        #profile = profile + 2*pi - max(profile)

        if not profile.size:
            continue

        profiles.append(profile)

        # chop off discontinous phase jumps
        if pruning:
            profile,offset = prune_pprofile(col, max_deriv = max_prune_deriv)
        
        # figure(1111)
        # plot(profile, label = j)
        # legend()
        
        mx = np.argmax(profile)

        ps1 = profile[:mx]
        ps2 = profile[mx:]

        if not ps1.size or not ps2.size:
            continue

        ts.append(tvec[j])
        phases_d.append(ps1)
        phases_up.append(ps2)
        q_d.append( (max(ps1) - min(ps1))/(2*pi) )
        q_up.append( (max(ps2) - min(ps2))/(2*pi) )

    if Plot:
        
        f = ppl.figure(11); f.clf()
        ppl.plot(ts,q_d,lw = 3,alpha = 0.4,label = 'lower')
        ppl.plot(ts,q_up,lw = 3,alpha = 0.4, label = 'upper')
        ppl.xlabel('time [min]')
        ppl.ylabel('q')
        if label:
            ppl.legend(frameon = False,loc = 'best')

    return np.array(ts), np.array(q_d), np.array(q_up)


def Plot_phase_profiles(pKymo, epochs = [],dt = 10, pin = True, skip = 0,ptitle = None,num = None):

    if type(pKymo) is not masked_array:
        print ('constructing empty mask..')
        pKymo = masked_array(data = pKymo, mask = np.zeros(pKymo.shape,dtype = bool))

    tvec = np.arange(0,pKymo.shape[1]*dt,dt)
    f = ppl.figure(num,figsize = (7,5))
    f.clf()
    ax = f.add_subplot(111)
    alphas = [1 -i/len(epochs) for i in range(len(epochs))] # not used

    colors = ['crimson','orange','darkseagreen','royalblue']
    assert (len(epochs) <= len(colors)),'only {} profiles possible..'.format(4)

    profiles = []
    for color,tind in zip(colors,epochs):

        p = pKymo[:,tind]
        vb = ~p.mask
        offset = 0
        if sum(vb) > 0:
            offset = np.where(vb)[0][0] # start of unmasked data


        # profile, offset = prune_pprofile(p)
        if pin:
            profile = np.unwrap(p[vb])
            profile = profile + 2*pi - max(profile)
        else:
            profile = p[vb]

        profiles.append(profile)
        
        label = 'time {}min'.format(tvec[tind] + skip*dt)
        ax.plot(np.arange(len(profile))+offset,profile,lw = 2,alpha = 0.7,label = label,color = color)
        
    ax.legend(frameon = False,loc = 'best',ncol = 2)
    if ptitle:
        title( ptitle )
    ax.set_xlabel('space [px]',fontsize = 24)
    ax.set_ylabel('phase [rad]',fontsize = 24)
    ax.set_yticks( [-pi,-pi/2,0,pi/2,pi,3/2*pi,2*pi] )
    ax.set_ylim( [-2.2*pi,2.2*pi] )
    ax.set_yticklabels(['$-\pi$','-$\pi/2$','0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])

    ppl.subplots_adjust(top = 0.92,bottom = 0.13)
        
    return profiles


def Plot_ampl_profiles(aKymo, epochs = [],dt = 10, pin = True, skip = 0,ptitle = None,num = None):

    if type(aKymo) is not masked_array:
        print ('constructing empty mask..')
        aKymo = masked_array(data = aKymo, mask = np.zeros(aKymo.shape,dtype = bool))

    tvec = np.arange(0,aKymo.shape[1]*dt,dt)
    f = ppl.figure(num,figsize = (7,5))
    f.clf()
    ax = f.add_subplot(111)
    alphas = [1 -i/len(epochs) for i in range(len(epochs))] # not used

    colors = ['crimson','orange','darkseagreen','royalblue']
    assert (len(epochs) <= len(colors)),'only {} profiles possible..'.format(4)

    profiles = []
    for color,tind in zip(colors,epochs):

        p = aKymo[:,tind]
        vb = ~p.mask
        offset = 0
        if sum(vb) > 0:
            offset = np.where(vb)[0][0] # start of unmasked data

        profile = p[vb]
            
        profiles.append(profile)
        
        label = 'time {}min'.format(tvec[tind] + skip*dt)
        ax.plot(np.arange(len(profile))+offset,profile,lw = 2,alpha = 0.7,label = label,color = color)
        
    ax.legend(frameon = False,loc = 'best',ncol = 2)
    if ptitle:
        title( ptitle )
    ax.set_xlabel('space [px]',fontsize = 24)
    ax.set_ylabel('amplitude [a.u.]',fontsize = 24)
    ax = ppl.gca()
    f.subplots_adjust(top = 0.92,bottom = 0.13,left = 0.18)
        
    return profiles


def Plot_freq_profiles(fKymo, epochs = [],dt = 10,skip = 0,under = -99,ptitle = None,num = None,win_len = 3):
    
    tvec = np.arange(0,fKymo.shape[1]*dt,dt)
    f = ppl.figure(num,figsize = (10,6))
    f.clf()
    ax = f.add_subplot(111)
    alphas = [1 -i/len(epochs) for i in range(len(epochs))] # not used

    colors = ['crimson','orange','darkseagreen','royalblue']
    assert (len(epochs) <= len(colors)),'only {} profiles possible..'.format(4)
        
    for color,tind in zip(colors,epochs):
        
        p = fKymo[:,tind]
        vb = ~p.mask
        offset = 0
        if sum(vb) > 0:
            offset = np.where(np.diff(vb))[0][0]

        profile = 1/p[vb]*60
        profile = smooth(profile,window_len = win_len)

        label = 'time {}min'.format(tvec[tind] + skip*dt)
        plot(np.arange(len(profile))+offset,profile,lw = 3,alpha = 0.7,label = label,color = color)
        ppl.legend(frameon = False,loc = 4,ncol = 2)
        if ptitle:
            title( ptitle )
        ax.set_xlabel('space [px]',fontsize = 24)
        ax.set_ylabel('frequency [1/h]',fontsize = 24)

    f.subplots_adjust(top = 0.98,bottom = 0.15)

    
        
def Plot_phase_surface(pkymo, dt = 10, under = -99):

    
    tvec = np.arange(0,pkymo.shape[1]*dt,dt)

    X,Y = meshgrid( np.arange(pkymo.shape[0]),np.arange(pkymo.shape[1]) )

    profiles = []
    ts = [] # time points for valid q-extraction
    zs = []
    for j,col in enumerate(pkymo.T):

        # z = under*ones(pkymo.shape[0])
        z = np.zeros(pkymo.shape[0])

        binds = col != under
        ms = np.where( np.diff(binds) )[0] # position of mask for every column
        
        if sum(binds) <= 3:
            continue

        profile = np.unwrap(col[binds])
        profiles.append(profile)

        mx = argmax(profile)
        profile = profile + 2*pi - max(profile)

        # print (ms)
        # fill old columns with unwrapped and shifted phases
        z[binds] = profile
        #z[:ms[0]] = min(profile[:mx])
        #z[ms[1]:] = min(profile[mx:])
        
        #plot(col)
        ts.append(tvec[j])
        zs.append(z)


    S = np.array( zs )
    X,Y = meshgrid( np.arange(S.shape[0],dtype = float),np.arange(S.shape[1],dtype = float) )

    #return X,Y,S
    # pk = ma.MaskedArray(pkymo, mask = mask)
    mask = S != 0
    print (X.shape,Y.shape,shape)

    cmap = ppl.get_cmap('jet')
    cmap.set_under('0.8')
    
    vmin = 1e-3
    print(S.shape)

    f = ppl.figure(13,figsize = (12,10))
    f.clf()
    ax = f.add_subplot(111, projection='3d')

    ax.set_title('Unwrapped phases')
    # col = time, row = space 
    ax.plot_surface(Y,X,S.T,cmap = cmap,vmin = vmin,rstride = 10,cstride = 8, lw = 0)
    
    #ax.plot_surface(Y,X,S.T,cmap = cmap,rstride = 20,cstride = 4, lw = 0) 
    #ax.plot_wireframe(Y,X,S.T,rstride = 20,cstride = 20) # col = space, row = time
    ax.set_zlim( (vmin,2.2*pi) )
    ax.set_xlabel('space [px]')
    ax.set_zlabel('phase [rad]')
    ax.set_ylabel('time [min]')
    ax.set_zticks([0,pi,2*pi])
    ax.set_zticklabels(['$0$','$\pi$','$2\pi$'])
    yticks = ax.get_yticks()
    ax.set_yticklabels(dt*yticks)
    ax.yaxis.labelpad = 30
    ax.xaxis.labelpad = 30
    ax.zaxis.labelpad = 30
    ax.tick_params( pad=10)

 
def Plot_Kymo(mat,dt = 10,aspect = None,cmap = 'afmhot', vmin = None,vmax = None,num = None,ptitle = None, cbar = False,offset = 0,bloc = None, interpolation = 'None', origin = None):

    fig,ax = prepare_kymo_plot(num, bloc = bloc)
    if aspect is None:
        aspect = mat.shape[1]/mat.shape[0]
    if type(cmap) is str:
        cmap = ppl.get_cmap(cmap)
        cmap.set_under('0.93')
        cmap.set_bad('0.93')
    im = ax.imshow(mat,aspect = aspect,cmap = cmap,vmin = vmin,vmax = vmax,interpolation = interpolation, origin = origin)
    if ptitle:
        ax.set_title(ptitle, fontsize = 22)
    if cbar:
        cb = colorbar(im,ax = ax,shrink = 0.7)
        #cb.set_label('Period [min]')

    # correct time labels
    xt = ax.get_xticks()
    ax.set_xticklabels( (xt*dt + offset*dt).astype(int))

        
# kind of a wrapper
def Plot_pKymo(mat,dt = 10,aspect = None,cmap = 'jet', num = None,ptitle = None, bloc = None):
    Plot_Kymo(mat,dt,aspect,cmap,vmin = -pi, vmax = pi, num = num, ptitle = ptitle, bloc = bloc) 

def prepare_kymo_plot(num = None, figsize = (8,8),bloc = None,ptitle = None):

    fig = ppl.figure(num,figsize = figsize)
    fig.clf()
    ax = ppl.gca()
    ax.set_xlabel('time [min]', fontsize = 23)
    ax.set_ylabel('space [px]', fontsize = 23)
    if ptitle:
        ax.set_title(ptitle, fontsize = 23)

    if not bloc:
        bloc = 25

    loc = ticker.MultipleLocator(base = bloc)
    ax.xaxis.set_major_locator(loc)
    loc = ticker.MultipleLocator(base = bloc/2.)
    ax.xaxis.set_minor_locator(loc) 
    ax.tick_params(labelsize = 20)
    fig.subplots_adjust(bottom = 0.15,left = 0.15)
        
    return fig,ax

def mk_intKymo_mask(iKymo, min_int = 1500,win_len = 41, two_sided = True,Plot = False, buf = 0):

    '''
    Use first maximum crossing the min. intensity threshold found by 
    zero-crossing of 1st derivative. 
    '''

    mask = np.zeros(iKymo.shape,dtype = bool)
    Nx = iKymo.shape[0]
    xv = np.arange(Nx)

    up = []
    down = []

    for i,profile in enumerate(iKymo.T):

        spl = spline(xv,profile, s = None)
        dspl = spl.derivative()
        dsig = dspl(xv)
        zero_crossings = np.where( np.diff( np.sign( dsig) ))[0]
        zero_crossings  = zero_crossings [np.where( profile[zero_crossings] > min_int)[0] ]

        #print (i,zero_crossings,profile[zero_crossings])
        # mask whole column
        if not np.size(zero_crossings):
            up.append(Nx)
            if two_sided:
                down.append(Nx/2)
            continue

        up_c = zero_crossings[0] - buf if zero_crossings[0] - buf >= 0 else 0 
        up.append(up_c)
        

        if two_sided:
            down_c = zero_crossings[-1] + buf if zero_crossings[-1] + buf < len(profile) else len(profile) - 1
            down.append(down_c)
            
    up = smooth(np.array(up),window_len = win_len)
    up = up.astype(int)
    for i,ind in enumerate(up):
        mask[:ind,i] = True

    if two_sided:
        down = smooth(np.array(down),window_len = win_len)
        down = down.astype(int)

        for i,ind in enumerate(down):
            mask[ind:,i] = True
            
    if Plot:
        Plot_Kymo(iKymo,num = 432,origin = None)
        ax = ppl.gca()
        ax.contour(mask,color = 'k')

    miKymo = masked_array(iKymo,mask)
    return miKymo



def mk_fKymo(miKymo,wAn,sigma_x = 1,sigma_t = 1, Plot = False):

    '''
    Periods from Wavelet transform on whole Kymograph, masking afterwards.
    '''

    matT = np.zeros(miKymo.shape)
    mask = miKymo.mask
    for i,int_vec in enumerate(miKymo):
        if sum(~int_vec.mask) < 10:
            matT[i,:] = 0
            continue

        # print(sum(~int_vec.mask))
        wAn.new_signal(int_vec)
        wAn.sinc_detrend()
        wAn.compute_spectrum(Plot = False)
        rdata = wAn.get_maxRidge(Thresh = 0)
        wav_Ts = rdata['periods']
        matT[i,:] = wav_Ts
        
    fKymo = masked_array(matT,mask = mask)

    if not Plot:
        return fKymo
    
    #------Plot-----------------------------------
    # gaussian blur
    fKymo = masked_array(matT,mask = miKymo.mask)
    gfKymo = gblur(fKymo,(sigma_x,sigma_t))
    gfKymo = masked_array(gfKymo,mask = miKymo.mask)
    
    fig,ax = prepare_kymo_plot(num = 467,figsize = (7,9),bloc = 25 )
    aspect = matT.shape[1]/matT.shape[0]
    cmap = ppl.get_cmap('plasma')
    cmap.set_under ('0.8')
    #im = imshow(gfKymo,aspect = aspect,vmin = wAn.periods[0],cmap = 'plasma')
    im = imshow(gfKymo,aspect = aspect,vmin = None,cmap = 'plasma')
    cb = colorbar(im,ax = ax,shrink = 0.9,orientation = 'horizontal')
    cb.set_label('Period [min]', fontsize = 26)
    xt = ax.get_xticks()
    ax.set_xticklabels( xt*wAn.dt )

    subplots_adjust(bottom = 0.05)
    return gfKymo


def Plot_per_profiles(mfKymo, epochs = [],dt = 10,offset = 0,under = -99,win_len = 3,ptitle = None,num = None):

    if type(mfKymo) is not masked_array:
        print ('constructing empty mask..')
        mfKymo = masked_array(data = mfKymo, mask = np.zeros(mfKymo.shape,dtype = bool))

    
    tvec = np.arange(0,mfKymo.shape[1]*dt,dt)
    f = ppl.figure(num,figsize = (10,6))
    f.clf()
    ax = f.add_subplot(111)
    alphas = [1 -i/len(epochs) for i in range(len(epochs))] # not used

    colors = ['crimson','orange','darkseagreen','royalblue']
    assert (len(epochs) <= len(colors)),'only {} profiles possible..'.format(4)
        
    for color,tind in zip(colors,epochs):
        
        p = mfKymo[:,tind]
        vb = ~p.mask
        offset = 0
        if sum(vb) > 0 and sum(vb) != mfKymo.shape[0]:
            offset = np.where(np.diff(vb))[0][0] # start of unmasked data

        profile = p[vb]

        profile = smooth(profile,window_len = win_len)

        label = 'time {}min'.format(tvec[tind] + offset*dt)
        plot(np.arange(len(profile))+offset,profile,lw = 3,alpha = 0.7,label = label,color = color)
        ppl.legend(frameon = False,loc = 'best',ncol = 2)
        if ptitle:
            title( ptitle )
        ppl.xlabel('space [px]',fontsize = 24)
        ppl.ylabel('period [min]',fontsize = 24)
        ax = ppl.gca()

    subplots_adjust(top = 0.98,bottom = 0.15)


def Plot_dphis(pKymo,mask):

    matPhi = np.zeros(pKymo.shape)
    for j,col in enumerate(pKymo.T):

        dp = np.diff(np.unwrap(col))
        matPhi[:-1,j] = dp*pKymo.shape[0]/2.
        
        matPhi[mask] = -99

    #------Plot-----------------------------------
    fig,ax = prepare_kymo_plot(figsize = (8,10))
    aspect = matPhi.shape[1]/matPhi.shape[0]
    im = ax.imshow(matPhi,aspect = aspect,vmin = -4*pi,vmax = 4*pi)
    ax.set_title("$\Delta \phi's$")

    cb = colorbar(im,ax = ax,shrink = 0.9, orientation = 'horizontal')
    cb.set_label('Phase gradient [rad/px]',fontsize = 26)
    cb.set_ticks([-4*pi,-2*pi,0,2*pi,4*pi])
    cb.set_ticklabels(['$-4\pi$','$-2\pi$','$0$','$2\pi$','$4\pi$'])

    xt = ax.get_xticks()
    ax.set_xticklabels( xt*dt )

    
    return matPhi


def read_image(fname,datadir):

    print('Reading {}'.format(fname))
    iKymo = io.imread(datadir + fname)
    print('Image dimensions:',iKymo.shape)

    return iKymo

    
#----------------------------------------
gblur = ndimage.filters.gaussian_filter
med = ndimage.filters.median_filter
#----------------------------------------

#=========Create default plots

def mk_tplot(num = 1, fig_size = (6,6), ylabel = r'Intensity $\frac{I}{I_0}$'):
    fig1 = ppl.figure(num,figsize = fig_size)
    fig1.clf()
    ax1 = ppl.gca()
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel(ylabel) # some latex moves :)
    #ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    fig1.subplots_adjust(bottom = 0.11,left = 0.17)

    return fig1, ax1




        
    
    
