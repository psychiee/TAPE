# -*- coding: utf-8 -*-
"""
04-chek_comps
 (1) READ the time-series photometry files (apx)
 (2) GENERATE the LC plots after filtering with SNR / dMAG

@author: wskang
@update: 2020/05/19
"""
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from photlib import read_params, prnlog

# READ the parameter file
par = read_params()

# PARAMETERS for the list of time-series observation
WORKDIR = par['WORKDIR']
LOGFILE = par['LOGFILE']
PHOT_APER = np.array(par['PHOTAPER'].split(','), float)   # Radius of aperture 
N_APER = len(PHOT_APER)
T_APER = int(par['APERUSED'])
TNUM1 = int(par['TARGETNUM'])
TNUM = TNUM1 - 1 # the index of target in the star list
TNAME = par['TARGETNAM']
WNAME = par['OBSDATE']+'-'+par['TARGETNAM']
CHKSIG, CHKDELM = float(par['CHKSIG']), float(par['CHKDELM'])

prnlog('#WORK: chek_comps')
prnlog(f'#WORKNAME: {WNAME}')
prnlog(f'#APERTURE: [{T_APER}] R={PHOT_APER[T_APER-1]}pix')
prnlog(f'#TARGETNUM: {TNUM1}')
prnlog(f'#LOGFILE: {LOGFILE}')
       
# MOVE to the working directory =======
CDIR = os.path.abspath(os.path.curdir)
os.chdir(WORKDIR)
#======================================

# READ the time-series log file 
flog = open(LOGFILE,'r')
lfrm, lname, lJD, lX, lFILTER = [], [], [], [], []
for line in flog:
    tmp = line.split()
    lfrm.append(int(tmp[0]))
    lname.append(tmp[1])
    lJD.append(float(tmp[2]))
    lX.append(float(tmp[4]))
    lFILTER.append(tmp[5])
FRM, FLIST, JD, X = np.array(lfrm), np.array(lname), np.array(lJD), np.array(lX)
flog.close()
JD0 = int(JD[0])

# PLOT the airmass variation 
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(JD-JD0, X,'r-',lw=2)
ax1.grid()
ax1.set_title(f'{WNAME} AIRMASS')
fig1.savefig(f'w{WNAME}-AIRMASS.png')
plt.close('all')

# CHECK number of frames and stars 
tmp = np.genfromtxt(FLIST[0]+'.apx') 
NSTARS = len(tmp[:,0])
NFRMS = len(FLIST)
FILTER = lFILTER[0]

# READ the apt files
FLX, ERR, MAG, MRR, FLG = [], [], [], [], []
for i, fidx in enumerate(FLIST):
    dat = np.genfromtxt(fidx+'.apx')
    FLX.append(dat[:,(T_APER+2)])
    ERR.append(dat[:,(N_APER+T_APER+2)])
    MAG.append(dat[:,(2*N_APER+T_APER+2)])
    MRR.append(dat[:,(3*N_APER+T_APER+2)])
    FLG.append(dat[:,(4*N_APER+4)])
# MAKE the arrays of photometric results by JDs and stars
# row: frames(time), col: stars(mag, flux) 
FLX, ERR = np.array(FLX), np.array(ERR)
MAG, MRR = np.array(MAG), np.array(MRR)
FLG = np.array(FLG)

# TARGET arrays
TFLX, TERR, TMAG, TMRR, TFLG = \
  FLX[:,TNUM], ERR[:,TNUM], MAG[:,TNUM], MRR[:,TNUM], FLG[:,TNUM]
NFRMS = len(FLX)
# LOOP for the comparisons
for j in range(1,len(FLX[0,:])+1): 

    # CHECK the index of comparisons
    cidx = j - 1
    if cidx == TNUM: continue

    x, y = JD-JD0, TMAG - MAG[:,cidx]
    yflg = TFLG + FLG[:,cidx]
    yerr = np.sqrt(TMRR**2 + MRR[:,cidx]**2)
    vv, = np.where(yflg == 0)
    VFRMS = len(vv)
    if VFRMS < NFRMS*0.5: 
        prnlog(f'{TNUM1:03d}-{j:03d}: [FAIL] need more points, {VFRMS:d} pts')
        continue

    # CALC. the statistics for the shift values
    ymed = np.median(y[vv])
    rms = np.sqrt(np.mean((ymed-y[vv])**2))
    ysig = np.std(y[vv])

    # CHECK the criteria for the comparisons
    if ysig > CHKSIG: 
        prnlog(f'{TNUM1:03d}-{j:03d}: [FAIL] SIG = {ysig:.2f}')
        continue 
    if ymed < -CHKDELM: 
        prnlog(f'{TNUM1:03d}-{j:03d}: [FAIL] DELM = {ymed:.2f}')
        continue

    prnlog(f'{TNUM1:03d}-{j:03d}: [OK] dM={ymed:.3f} RMS={rms:.3f}')

    # PLOT the light curves of magnitude for each star  
    fig1, ax1 = plt.subplots(num=99,figsize=(10,5))
    ax1.errorbar(x[vv],y[vv],yerr=yerr[vv], fmt='ko', ms=5, alpha=0.5, label=f'rms={rms:.5f}')
    cc, = np.where((ymed - y[vv])**2 > 4*ysig**2)
    for ix, iy, ifrm in zip(x[vv[cc]],y[vv[cc]],FRM[vv[cc]]):
        ax1.text(ix,iy,'%d' % (ifrm,),color='r',fontsize=10)
    ax1.set_ylim(ymed+0.03,ymed-0.03)
    ax1.set_title(f'{WNAME} Light Curve (MAG-{j:03d}) JD0={JD0:d}')
    ax1.grid()
    ax1.legend()
    fig1.savefig(f'w{WNAME}-{FILTER}-CHK-{j:03d}.png')
    fig1.clf()

plt.close('all')

# RETURN to the directory ===========
os.chdir(CDIR) 
#====================================