##################################################################################
# Tools for time-frequency analysis with Morlet Wavelets
# Inspired by 'A Practical Guide to Wavelet Analysis' from Torrence and Compo 1998
# and 'Identification of Chirps with Continuous Wavelet Transform' from Carmona,Hwang and Torresani 1995
#
# Version 0.4 Januar 2018, Gregor Moenke (gregor.moenke@embl.de)
##################################################################################


from __future__ import division,print_function
import os,sys
import pylab as ppl
import numpy as np
from numpy import pi, e, cos, sin, sqrt
from numpy.random import randn
from os import path,walk
from scipy.optimize import leastsq
from scipy.signal import hilbert,cwt,ricker,lombscargle,welch,morlet,bartlett
import pandas as pd

from matplotlib import rc

# rc('font', family='sans-serif', size = 18)
# rc('lines', markeredgewidth = 0)


# global variables
#-----------------------------------------------------------
# thecmap = 'plasma' # the colormap for the wavelet spectra
thecmap = 'viridis' # the colormap for the wavelet spectra
omega0 = 2*pi # central frequency of the mother wavelet
ridge_def_dic = {'Temp_ini' : 0.2, 'Nsteps' : 25000, 'max_jump' : 3, 'curve_pen' : 0.2, 'sub_s' : 2, 'sub_t' : 2} # default dictionary for ridge detection by annealing
xi2_95 = 5.99
xi2_99 = 9.21
#-----------------------------------------------------------


class TFAnalyser:

    def __init__(self,periods,dt,T_cut_off, vmax = 20, offset = 0):

        self.periods = periods
        self.dt = dt
        self.T_c = T_cut_off
        self.vmax = vmax
        self.offset = offset

        self._has_spec = False
        self._has_dsignal = False
        self._has_signal = False        
        self._has_ridge = False
        self._has_results = False
        
        self.ax_spec = None
        self.signal = None
        self.dsignal = None
        self.name = ''

    def new_signal(self,raw_signal, name = ''):

        self.signal = raw_signal
        self.name = str(name)
        self.tvec = np.arange(0,len(raw_signal)*self.dt,self.dt) + self.offset
        #print(len(self.tvec),len(raw_signal))

                       
        if not self._has_results:

            results=pd.DataFrame(index = range(len(raw_signal))) # initialize DataFrame index             
            results.insert(0,'Time (min)',self.tvec) # assign an extra column in the front

            self.results = results
            self._has_results = True
            
        self._has_spec = False
        self._has_dsignal = False
        self._has_ridge = False
        self.ridge_data = None
        self.ax_spec = None
        
        self._has_signal = True

        
    def compute_spectrum(self, raw_signal = None, Plot = True, time_label = 'min',fig_num = None, ptitle = None, detrend = True, coi = True):

        if raw_signal is not None:
            self.new_signal(raw_signal)
        if detrend:
            self.sinc_detrend()
            signal = self.dsignal
        else:
            signal = self.signal
        
        # easy 
        dt = self.dt
        periods = self.periods
        vmax = self.vmax

        '''

        Computes the Wavelet spectrum for a given *signal* for the given *periods*
        
        signal  : a sequence
        the time-series to be analyzed, detrend beforehand!
        dt      : the sampling interval scaled to desired time units
        periods : the list of periods to compute the Wavelet spectrum for, 
              must have same units as dt!

        vmax       : Maximum power for z-axis colormap display, if *None* scales automatically
        
        Plot       : set to False if no plot is desired
        time_label : the label for the time unit when plotting
        fig_num    : the figure number when plotting, if *None* a new figure will be created

        returns:

        wlet : the Wavelet transform with dimensions len(periods) x len(signal) 
        
        '''

        if periods[0] < 2*dt:
            print()
            print('Warning, Nyquist limit is',2*dt,time_label,'!!')
            print()

        signal = np.array(signal)
        periods = np.array(periods)
        dt = float(dt)
        sfreq = 1/dt # the sampling frequency
        tvec = np.arange(0,len(signal)*dt+dt,dt) + self.offset

        Nt = len(signal) # number of time points

        #--------------------------------------------
        scales = scales_from_periods(periods,sfreq,omega0)
        #--------------------------------------------
        coi_m = Morlet_COI(periods,sfreq,omega0) # slope of COI

        #mx_per = 4*len(signal)/((omega0+sqrt(2+omega0**2))*sfreq)
        mx_per = dt*len(signal)
        if max(periods) > mx_per:

            print()
            print ('Warning: Very large periods chosen!')
            print ('Max. period should be <',print(mx_per),time_label)
            print ('proceeding anyways...')

        Morlet = mk_Morlet(omega0)
        wlet = CWT(signal, Morlet, scales) #complex wavelet transform
        sig2 = np.var(signal)
        modulus = np.abs(wlet)**2/sig2 # normalize with variance of signal

        if Plot:

            ax = _plot_modulus(modulus,periods,dt,offset = self.offset,vmax = self.vmax,fig_num=fig_num,ptitle = ptitle, time_label = time_label)

            if coi:
                N_2 = int(len(signal)/2.)
                ax.plot(tvec[:N_2+1],coi_m*tvec[:N_2+1],'k-.',alpha = 0.3)
                ax.plot(tvec[N_2:],tvec[-1]*coi_m - coi_m*tvec[N_2:],'k-.',alpha = 0.3)
                print(tvec)
                print(N_2,tvec[N_2:])


        self.wlet = wlet
        self.modulus = modulus
        self.ax_spec = ppl.gca()
        self._has_spec = True

    def get_maxRidge(self, Thresh = 0, smoothing = True):

        '''
        Computes the ridge as consecutive maxima of the modulus.

        Returns the ridge_data dictionary (see mk_ridge_data)!

        '''


        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        # for easy integration
        modulus = self.modulus

        Nt = modulus.shape[1] # number of time points

        #================ridge detection============================================

        # just pick the consecutive modulus (squared complex wavelet transform) maxima as the ridge

        ridge_y = np.array( [np.argmax(modulus[:,t]) for t in np.arange(Nt)] ,dtype = int)
        self._has_ridge = True
        rd = self.mk_ridge_data(ridge_y,Thresh = Thresh)
        self.ridge_data = rd
        
        return rd

    def draw_maxRidge(self, Thresh = 0, smoothing = True, color = 'orangered'):

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        rdata = self.get_maxRidge(Thresh,smoothing)

        if rdata is None:
            return

        self.ax_spec.plot(rdata['time'],rdata['periods'],'o',color = color,alpha = 0.5,ms = 3)


    def get_annealRidge(self,period_guess, pars = ridge_def_dic):

        # for easy integration
        modulus = self.modulus
        wlet = self.wlet
        dt = self.dt
        periods = self.periods
        tvec = self.tvec

        # get modulus index of initial straight line ridge
        y0 = np.where(periods < period_guess)[0][-1]
        
        ridge_y, F_c = find_ridge_anneal(modulus, y0, pars['Temp_ini'],pars['Nsteps'],pars['max_jump'], pars['curve_pen'])

        self._has_ridge = True
        rd = self.mk_ridge_data(ridge_y)
        self.ridge_data = rd

        self.anneal_ridge = ridge_y
        
        return rd

    def draw_Ridge(self, color = 'orangered'):

        if not self._has_ridge:
            print('Need to compute a ridge first..!')
            return

        rdata = self.ridge_data
        self.ax_spec.plot(rdata['time'],rdata['periods'],'o',color = color,alpha = 0.5,ms = 3)

    def mk_ridge_data(self,ridge_y, Thresh = 0, smoothing = True, win_len = 17):

        
        '''

        Given the ridge coordinates, returns a dictionary containing:

        periods  : the instantaneous periods from the ridge detection    
        time     : the t-values of the ridge
        z        : the (complex) z-values of the Wavelet along the ridge
        phases   : the arg(z) values
        power    : the amplitude |z|

        Moving average smoothing of the ridge supported.

        '''

        if not self._has_ridge:
            print('Need to detect a ridge first..!')
            return
        
        # for easy integration
        modulus = self.modulus
        wlet = self.wlet
        dt = self.dt
        periods = self.periods
        tvec = self.tvec


        Nt = modulus.shape[1] # number of time points

        ridge_maxper = periods[ridge_y]
        ridge_z = wlet[ ridge_y, np.arange(Nt) ] # picking the right t-y values !

        ridge_power = modulus[ridge_y, np.arange(Nt)]

        inds = ridge_power > Thresh # boolean array of positions of significant oscillations
        sign_maxper = ridge_maxper[inds] # periods which cross the power threshold
        ridge_t = tvec[inds]
        ridge_phi = np.angle(ridge_z)[inds]

        if (sum(inds)) < 1: 
            print( 'Can not identify ridge, no significant oscillations found, check spectrum/threshold!')
            return None

        if smoothing is True:

            if (sum(inds)) < win_len: # ridge smoothing window len
                print( 'Can not identify ridge, no significant oscillations found, check spectrum/threshold!')
                return None

            sign_maxper = smooth(ridge_maxper,win_len)[inds] # smoothed maximum estimate of the whole ridge..


        ridge_data = {'periods' : sign_maxper, 'time' : ridge_t, 'z' : ridge_z, 'power' : ridge_power, 'inds' : inds, 'phase' : ridge_phi}

        MaxPowerPer=ridge_maxper[np.nanargmax(ridge_power)]  # period of highest power on ridge

        print('Period with max power of {:.2f} is {:.2f}'.format(np.nanmax(ridge_power),MaxPowerPer)) 
        # ridge_data = self.ridge_data

        return ridge_data

        

    def get_mean_spectrum(self):

        ''' Average over time '''

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        mfourier = np.sum(self.modulus,axis = 1)/len(self.signal)

        return mfourier

        
    def draw_AR1_confidence(self,alpha):

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        x,y = np.meshgrid(self.tvec,self.periods) # for plotting the wavelet transform
        
        ar1power = ar1_powerspec(alpha,self.periods,self.dt)
        conf95 = xi2_95/2.
        conf99 = xi2_99/2.
            
        scaled_mod = np.zeros(self.modulus.shape)

        # maybe there is a more clever way
        for i,col in enumerate(self.modulus.T):
            scaled_mod[:,i] = col/ar1power
            
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_95/2.],linewidths = 1.5,colors = '0.95')
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_99/2.],linewidths = 1.5,colors = 'orange')


        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(wlet.shape)) # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(wlet.shape)) # should be ~0.01
        
    def save_ridge(self):

        if not self._has_ridge:
            print()
            print('No ridge analysis found, can not write any new results ..')
            print()
            return

        r = self.ridge_data['periods']
        t = self.ridge_data['time']
        power = self.ridge_data['power']
        phases = self.ridge_data['phase']
        inds = self.ridge_data['inds']
        index = np.arange(len(self.signal))
    
        s1 = pd.Series(data = r, index = index[inds])
        self.results['Periods ' + self.name] = s1 # add a column with the smoothed_maxpers to the dataframe
        s2 = pd.Series(data = power, index = index)
        
        self.results['RidgePower ' + self.name]= s2 # add a column with the ridge powers to the dataframe
        s3 = pd.Series(data = phases, index = index)
        self.results['Phases ' + self.name] = s3 # add a column with the smoothed_maxpers to the 

    def export_results(self,outname):

        if not self._has_results:
            print()
            print('No results to export yet..')
            print()
            return

        self.results.to_excel(outname+'.xlsx',index=False,header=True)
        print('Wrote {}.xlsx'.format(outname))

        
    def get_trend(self):

        trend = sinc_smooth(self.signal,self.T_c,self.dt)
        
        return trend

    def sinc_detrend(self):

        if not self._has_signal:
            print()
            print('No input signal..exiting!')
            print()
            return

        
        detrended = sinc_smooth(self.signal,self.T_c,self.dt,detrend = True)

        self._has_dsignal = True
        self.dsignal = detrended
        
        return detrended

    def plot_signal(self,with_trend = True, fig_num = 1, ptitle = None,time_label = 'min'):
        
        if not self._has_signal:
            print()
            print('No input signal found..exiting!')
            print()
            return
        
        if with_trend:
            trend = self.get_trend()

        dt = self.dt
        
        tvec = self.tvec
        
        fsize = (8,6)
        fig1 = ppl.figure(fig_num,figsize = fsize)
        fig1.clf()
        ax1 = ppl.gca()

        if ptitle:
            ax1.set_title(ptitle)
        ax1.plot(tvec,self.signal,lw = 1.5, color = 'royalblue',alpha = 0.8)
        if with_trend:
            ax1.plot(tvec,trend,color = 'orange',lw = 1.5) # plot the trend
        ax1.set_xlabel('Time [' + time_label + ']')
        ax1.set_ylabel(r'Intensity $\frac{I}{I_0}$') # some latex moves :)
        ppl.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
        fig1.subplots_adjust(bottom = 0.11,left = 0.17)


    def plot_detrended(self, fig_num = 2, ptitle = None,time_label = 'min'):
            
        if not self._has_dsignal:
            print()
            print('No detrended signal found..exiting!')
            print()
            return

        dt = self.dt
        
        tvec = self.tvec
        
        fsize = (8,6)

        fig1 = ppl.figure(fig_num,figsize = fsize)
        fig1.clf()
        ax1 = ppl.gca()

        if ptitle:
            ax1.set_title(ptitle)
            
        ax1.plot(tvec,self.dsignal,lw = 1.5, color = 'royalblue',alpha = 0.8)
        ax1.set_xlabel('Time [' + time_label + ']')
        ax1.set_ylabel(r'Intensity $\frac{I}{I_0}$') # some latex moves :)
        ppl.ticklabel_format(style='sci',axis='y',scilimits=(0,0))             
        fig1.subplots_adjust(bottom = 0.13,left = 0.17)


#---------Annealing helper routines------------------------------------------------

def find_ridge_anneal(landscape,y0,T_ini,Nsteps,mx_jump = 2,curve_pen = 0):

    ''' 
    Taking an initial straight line guess at *y0* finds a ridge in *landscape* which 
    minimizes the cost_func_anneal by the simulated annealing method.

    landscape - time x scales signal representation (modulus of Wavelet transform)
    y0        - initial ridge guess is straight line at scale landscape[y0] 
                -> best to set it close to a peak in the Wavelet modulus (*landscape*)
    T_ini     - initial value of the temperature for the annealing method
    Nsteps    - Max. number of steps for the algorithm
    mx_jump   - Max. distance in scale direction covered by the random steps
    curve_pen - Penalty weight for the 2nd derivative of the ridge to estimate -> 
                high values lead to  less curvy ridges

    '''

    print()
    print('started annealing..')
    
    incr = np.arange(-mx_jump,mx_jump+1) #possible jumps in scale direction
    incr = incr[incr!=0] #remove middle zero
    
    Nt = landscape.shape[-1] # number of time points
    Ns = landscape.shape[0] # number of scales
    t_inds = np.arange(Nt)
    ys = y0*ones(Nt,dtype = int) #initial ridge guess is straight line at scale landscape[y0]

    Nrej = 0

    T_k = T_ini/10. # for more natural units
    for k in range(Nsteps):
        
        F = cost_func_anneal(ys,t_inds,landscape,0,curve_pen)
        
        pos = randint(0,len(ys),size = 1) # choose time position to make random scale jump

        # dealing with the scale domain boundaries
        if ys[pos] >= Ns-mx_jump-1:
            eps = -1

        elif ys[pos] < mx_jump :
            eps = +1

        # jump!
        else:
            eps = choice(incr,size = 1)
            
        ys[pos] = ys[pos] + eps # the candidate
            
        F_c = cost_func_anneal(ys,t_inds,landscape,0,curve_pen)
        
        accept = True
        
        # a locally non-optimal move occured
        if F_c > F:
            u = uniform()
            
            # reject bad move? exp(-(F_c - F)/T_k) is (Boltzmann) probability for bad move to be accepted
            if u > np.exp(-(F_c - F)/T_k):
                accept = False

        if not accept:
            ys[pos] = ys[pos] - eps #revert the wiggle
            Nrej += 1

        if accept:
            Nrej = 0

                
        T_k = T_ini/log(2+k)/10. # update temperature

    print()
    print('annealing done!')
    print('final cost:',F_c)
    print('number of final still steps:',Nrej)
    return ys,F_c

def cost_func_anneal(ys,t_inds,landscape,l,m):

    '''
    Evaluates ridge candidate *ys* on *landscape* plus penalizing terms
    for 1st (*l*) and 2nd (*m*) derivative of the ridge curve.
    '''

    N = len(ys)
    D = -sum(landscape[ys,t_inds])
    S1 = l*sum(abs(np.diff(ys,1)))
    S2 = m*sum(abs(np.diff(ys,2)))

    #print D,S1,S2,D + S1 + S2
    
    return (D + S1 + S2)/N

        


#--------------------the general plotting routines--------------------------------------

def Plot_signal(signal,dt,fig_num = None, time_label = 'min', fsize = (8,4)):
    
    tvec = np.arange(0,len(signal)*dt,dt)
    

    fig1 = ppl.figure(fig_num,figsize = fsize)
    fig1.clf()
    ax1 = ppl.gca()

    ax1.plot(tvec,signal,lw = 2., color = 'royalblue',alpha = 0.8)
    ax1.set_xlabel('Time [' + time_label + ']')
    ax1.set_ylabel('Signal')
    fig1.subplots_adjust(bottom = 0.2)

    return ax1

def _plot_modulus(modulus,periods,dt,offset = 0,vmax = None, fig_num = None, ptitle = None,time_label = 'min'):

    tvec = np.arange(0,modulus.shape[1]*dt,dt) + offset
    
    x,y = np.meshgrid(tvec,periods) # for plotting the wavelet transform
    
    fsize = (8,7)
    fig1 = ppl.figure(fig_num,figsize = fsize)
    fig1.clf()
    ax1 = ppl.gca()

    #im = ax1.pcolor(x,y,modulus,cmap = thecmap,vmax = vmax)
    aspect = len(tvec)/len(periods)
    im = ax1.imshow(modulus[::-1],cmap = thecmap,vmax = vmax,extent = (tvec[0],tvec[-1],periods[0],periods[-1]),aspect = 'auto')
    ax1.set_ylim( (periods[0],periods[-1]) )
    ax1.set_xlim( (tvec[0],tvec[-1]) )
    if ptitle:
        ax1.set_title(ptitle)

 
    cb = ppl.colorbar(im,ax = ax1,orientation='horizontal',fraction = 0.08,shrink = 1.)
    cb.set_label('$|\mathcal{W}_\Psi(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 23)

    ax1.set_xlabel('Time [' + time_label + ']')
    ax1.set_ylabel('Period [' + time_label + ']')
    ppl.subplots_adjust(bottom = 0.11, right=0.95,left = 0.15,top = 0.95)

    return ax1

#------------------------------------------------------------------------------------------


def ar1_powerspec(alpha,periods,dt):
    res = (1-alpha**2)/(1+alpha**2 - 2*alpha*cos(2*pi*dt/periods))

    return res

def ar1_sim(alpha,sigma,N,x0 = None):

    N = int(N)
    sol = np.zeros(N)

    if x0 is None:
        x0 = randn()
        
    sol[0] = x0

    for i in range(1,N):
        sol[i] = alpha*sol[i-1] + sigma*randn()

    return sol


# vectorial mean -> 2nd Order parameter
def mean_phase(phis):

    my = sum( [sin(phi) for phi in phis] )
    mx = sum( [cos(phi) for phi in phis] )

    return np.arctan2(my,mx) # phi = tan(y/x)

# 1st order par
def order_par(thetas):
    N = len(thetas)
    x_tot = sum(cos(thetas))/N
    y_tot = sum(sin(thetas))/N
    
    return norm( np.array( (x_tot,y_tot) ) )


# difference of phases on the unit circle
def phase_diff(phi1,phi2):
    delta1 = 2*pi - phi1 # rotate reference frame
    sp2 = phi2 + delta1 

    return np.arctan(sin(sp2),cos(sp2))

#===============Filter===Detrending==================================

def smooth(x,window_len=11,window='bartlett',data = None):
    """smooth the data using a window with requested size.

    input:
    x: the input signal
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    flat window will produce a moving average smoothing.
    data: if not None, will be used as evaluated window!

    """

    x = np.array(x)

    # use externally derieved window evaluation
    if data is not None:
        window_len = len(data)
        window = 'extern'

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        raise ValueError("window must not be shorter than 3")

    if window_len%2 == 0:
        raise ValueError("window_len should be odd")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','triang','extern']:
       raise ValueError("Window is none of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','triang','extern'")

   
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
                                        #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')

    elif window == 'triang':
        w = triang(window_len)

    elif window == 'extern':
        w = data
        
    else:
        w=eval(window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int((window_len-1)/2):len(y)-int((window_len-1)/2)]


def sinc_filter(M, f_c = 0.2):

    ''' 
    Cutoff frequency f_c in sampling frequency unit, max 0.5!
    M is blackman window length and must be even, output length will be M+1.

    '''

    # not very effective, but should be get called only once per convolution

    assert M%2 == 0,'M must be even!'
    res = []

    for x in np.arange(0,M+1):
            
        if x == M/2:
            res.append(2*pi*f_c)
            continue
    
        r = sin(2*pi*f_c*(x - M/2))/( x - M/2 ) # the sinc filter unwindowed
        r = r * (0.42 - 0.5*cos(2*pi*x/M) + 0.08*cos(4*pi*x/M)) # blackman window
        res.append(r)

    res = np.array(res)
    res = res/sum(res)
            
    return res


def sinc_detrend(raw_signal,T_c,dt):

    signal = np.array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt/T_c
    M = len(signal) - 1 # max for sharp roll-off

    # M needs to be even
    if M%2 != 0:
        M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data = w)
    sinc_detrended = signal - sinc_smoothed

    return sinc_detrended

def sinc_smooth(raw_signal,T_c,dt,M = None,detrend = False):

    signal = np.array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt/T_c # max T_c = 2*dt -> max f_c = 0.5 (Nyquist)

    if M is None:
        
        M = len(signal) - 1 # max for sharp roll-off

        # M needs to be even
        if M%2 != 0:
            M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data = w)

    if detrend:
        sinc_smoothed = signal - sinc_smoothed

    return sinc_smoothed


def detrend(raw_signal,winsize = 7,window = 'flat', data = None):

    avsignal = smooth(raw_signal,winsize,window = window, data = data) 
    dsignal = raw_signal - avsignal             # detrend by subtracting filter convolution

    return dsignal

#=============WAVELETS===============================================================

def scales_from_periods(periods,sfreq,omega0):
    scales = (omega0+sqrt(2+omega0**2))*periods*sfreq/(4*pi) #conversion from periods to morlet scales
    return scales

def Morlet_COI(periods,sfreq,omega0):
    # slope of Morlet e-folding time in tau-periods (spectral) view
    m= 4*pi/(np.sqrt(2)*(omega0+sqrt(2+omega0**2)))
    return m

# is normed to have unit energy on all scales! ..to be used with CWT underneath
def mk_Morlet(omega0):

    def Morlet(t,scale):
        res = pi**(-0.25)*np.exp(omega0*1j*t/scale)*np.exp(-0.5*(t/scale)**2)
        return 1/sqrt(scale)*res
    
    return Morlet

# allows for complex wavelets, needs scales scaled with sampling freq!
def CWT(data,wavelet,scales):

    # test for complexity
    if np.iscomplexobj( wavelet(10,1) ):
        output = np.zeros([len(scales), len(data)],dtype = complex)
    else:
        output = np.zeros([len(scales), len(data)])

    vec = np.arange(-len(data)/2, len(data)/2) # we want to take always the maximum support available
    for ind, scale in enumerate(scales):
        wavelet_data = wavelet( vec, scale)
        output[ind, :] = np.convolve(data, wavelet_data,
                                  mode='same')
    return output


# ================= Helper functions ================================================

def read_csv(file_name):
    
    df=pd.read_csv(file_name)
    
    Ncols=df.shape[1]   # number of columns
    data=[]
    headers=[]
    for N in range(1,Ncols):    # skip the first as this contains the measurement time points
        raw_col=df.iloc[:,N]
        col=raw_col.tolist()        
        header=df.columns[N]
        data.append(col)
        headers.append(header)
        
    print('Read in',len(data),' columns from',file_name)
    return data, headers


def read_xls(file_name,horizontal = False):

    '''
    Reads in single sheet Excel files (.xls and .xlsx) and returns a 
    list of numerical sequences for every horizontal OR vertical data in the input sheet.
    Mixed type of data columns and rows in one sheet is not supported.

    arguments:
    ----------

    file_name  : string
                 the input file name as string
    horizontal : boolean
                 set to True if data is organized in rows (left-right), 
                 leave False if you have columns (top-down)
    '''

    num_types = (int,float)

    sheet_index = 0

    book = open_workbook(file_name)
    sheet = book.sheets()[sheet_index]

    data = []

    if horizontal:
        N = sheet.nrows
        retr = sheet.row_values # retrieve data rows
        direct = 'rows'

    else:
        N = sheet.ncols
        retr = sheet.col_values # retrieve data columns
        direct = 'columns'

    for n in range(N):

        raw_col = retr(n)

        # filter out non_numbers
        col = [val for val in raw_col if type(val) in num_types]

        # discard empty lists
        if col:
            data.append(col)
            
    print('Read in',len(data),direct,'from',file_name)
    
    return data


if __name__ == '__main__':



    T = 10
    dt = T/20.
    ps = np.linspace(5*dt,2*T,250)
    tvec = np.arange(0,3.3*T,dt)
    s1 = cos(tvec*2*pi/T )
    s1a = s1 + 10
    s2 = np.exp(-(tvec - np.mean(tvec))**2/np.mean(tvec)**2)*sin(tvec*2*pi/T)
    wAn = TFAnalyser(periods = ps, dt = dt, T_cut_off = 250)

    wAn.compute_spectrum(s1a)
    print('created signal s1, s2 and analyzer wAn for testing')

