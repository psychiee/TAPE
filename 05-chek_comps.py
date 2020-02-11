# -*- coding: utf-8 -*-
"""
05-chek_comps
 (1) READ the time-series data files (apx)
 (2) GENERATE the LC plots after filtering with SNR 

@author: wskang
@update: 2019/09/25
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from photlib import read_params

par = read_params()
os.chdir(par['WORKDIR'])
LOGFILE = par['LOGFILE']
PHOT_APER = np.array(par['PHOTAPER'].split(','), float)   # Radius of aperture 
N_APER = len(PHOT_APER)
T_APER = int(par['APERUSED'])
TNUM1 = int(par['TARGETNUM'])
TNAME = par['TARGETNAM']
OBSDATE, TARGET = par['OBSDATE'], par['TARGETNAM']
CHKSIG, CHKDELM = float(par['CHKSIG']), float(par['CHKDELM'])

TNUM = TNUM1 - 1 
# READ the time-series log file 
flog = open(LOGFILE,'r')
lfrm, lname, lJD, lX = [], [], [], []
for line in flog:
    tmp = line.split()
    lfrm.append(int(tmp[0]))
    lname.append(tmp[1])
    lJD.append(float(tmp[2]))
    lX.append(float(tmp[4]))
FRM, FID, JD, X = np.array(lfrm), np.array(lname), np.array(lJD), np.array(lX)
flog.close()
JD0 = int(JD[0])
# PLOT the light curves of magnitude for each star  
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(JD-JD0, X,'r-',lw=2)
ax1.grid()
ax1.set_title(OBSDATE+'-'+TNAME+' AIRMASS')
fig1.savefig('w'+OBSDATE+'-'+TNAME+'-AIRMASS')
plt.close('all')
# CHECK number of frames and stars 
tmp = np.genfromtxt(FID[0]+'.apx') 
NSTARS = len(tmp[:,0])
NFRMS = len(FID)

# READ the apt files
FLX, ERR, MAG, MRR, FLG = [], [], [], [], []
for i, fidx in enumerate(FID):
    dat = np.genfromtxt(fidx+'.apx')
    FLX.append(dat[:,(T_APER+2)])
    ERR.append(dat[:,(N_APER+T_APER+2)])
    MAG.append(dat[:,(2*N_APER+T_APER+2)])
    MRR.append(dat[:,(3*N_APER+T_APER+2)])
    FLG.append(dat[:,(4*N_APER+4)])
    print ('Read .apx file of frame#', FRM[i])
# MAKE the arrays of photometric results by JDs and stars
# row: frames(time), col: stars(mag, flux) 
FLX, ERR = np.array(FLX), np.array(ERR)
MAG, MRR = np.array(MAG), np.array(MRR)
FLG = np.array(FLG)

# TARGET arrays
TFLX, TERR, TMAG, TMRR, TFLG = \
  FLX[:,TNUM], ERR[:,TNUM], MAG[:,TNUM], MRR[:,TNUM], FLG[:,TNUM]
NFRMS = len(FLX) 
for j in range(1,len(FLX[0,:])+1): 
    
    cidx = j - 1
    if cidx == TNUM: continue

    x, y = JD-JD0, TMAG - MAG[:,cidx]
    yflg = TFLG + FLG[:,cidx]
    yerr = np.sqrt(TMRR**2 + MRR[:,cidx]**2)
    vv, = np.where(yflg == 0)
    VFRMS = len(vv)
    if VFRMS < NFRMS*0.5: 
        print ('%03i Star: need more points, %i pts' % (j, VFRMS,))
        continue
    
    ymed = np.median(y)
    rms = np.sqrt(np.mean((ymed-y)**2))
    ysig = np.std(y)

    if ysig > CHKSIG: 
        print ('%03i Star: SIG = %.2f' % (j,ysig,))
        continue 
    if ymed < -CHKDELM: 
        print ('%03i Star: DELM = %.2f' % (j,ymed,))
        continue
    
    print ('%03i Star: dM=%.3f RMS=%.3f' % (j,ymed,rms))
    # PLOT the light curves of magnitude for each star  
    fig1, ax1 = plt.subplots(figsize=(10,5))
    
    ax1.errorbar(x[vv],y[vv],yerr=yerr[vv],fmt='ko',\
                 ms=5, alpha=0.5, label='rms=%.5f' % (rms,))
    cc, = np.where((ymed - y)**2 > 4*rms**2)     
    for ix, iy, ifrm in zip(x[cc],y[cc],FRM[cc]):
        ax1.text(ix,iy,'%d' % (ifrm,),color='r',fontsize=10)
    ax1.set_ylim(ymed+0.03,ymed-0.03)
    ax1.set_title(TNAME+' Light Curve (MAG-%03d) JD0=%d' % (j,JD0))
    ax1.grid()
    ax1.legend()
    fig1.savefig('w'+TNAME+'-TEST-%03d' % (j,))
    fig1.clf()
    

