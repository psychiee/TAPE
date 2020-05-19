# -*- coding: utf-8 -*-
"""
05-plot_lightcurve
 (1) READ the time-series data files (apx)
 (2) GENERATE the LC plots with the comparisons
 (3) PLOT the compare btw. comparisons 
 (4) GENERATE the LC data files 

@author: wskang
@update: 2020/05/19
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from photlib import read_params, prnlog, sigma_clip

par = read_params()

# MOVE to the working directory
os.chdir(par['WORKDIR'])

#==================================================================
# TARGET information 
#==================================================================
PHOT_APER = np.array(par['PHOTAPER'].split(','), float)   # Radius of aperture 
N_APER = len(PHOT_APER)
T_APER = int(par['APERUSED']) # index of used aperture
TNUM1 = int(par['TARGETNUM'])  # Index of the target for variability
OBSDATE, TARGET = par['OBSDATE'], par['TARGETNAM']  # Obs. info. 
WNAME = OBSDATE+'-'+TARGET  # file name to save data 
CLIST1 = np.array(par['COMPNUMS'].split(','), int) # Indices of the comparisons
LOGFILE = par['LOGFILE']
#==================================================================
prnlog('#WORK: plot_lightcurve')
prnlog('#WORKNAME: {}'.format(WNAME))
prnlog('#APERTURE: [{}] {} pix'.format(T_APER, PHOT_APER[T_APER-1]))
prnlog('#TARGET: {}'.format(TARGET))
prnlog('#TARGET INDEX: {}'.format(TNUM1))
prnlog('#COMPARISON INDICES: {}'.format(CLIST1))
prnlog('#LOGFILE: {}'.format(LOGFILE))

# CONVERT number into index
TNUM = TNUM1 - 1

# READ the time-series log file 
flog = open(LOGFILE,'r')
lfrm, lname, lJD, lFILTER = [], [], [], [] 
for line in flog:
    tmp = line.split()
    lfrm.append(int(tmp[0]))
    lname.append(tmp[1])
    lJD.append(float(tmp[2]))
    lFILTER.append(tmp[5])
FRM, FLIST, JD = np.array(lfrm), np.array(lname), np.array(lJD)
#FILTER = lFILTER[0]
flog.close()

# SHOW the first image 
hdu = fits.open(FLIST[0]+'.fits')
FILTER = hdu[0].header['FILTER']
img = hdu[0].data
ny, nx = img.shape
iavg, imed, istd = sigma_clip(img)
fig, ax = plt.subplots(num=1, figsize=(9,9), dpi=100)
z1, z2 = imed, imed+istd*8
ax.imshow(-img, vmin=-z2, vmax=-z1, cmap='gray')

# READ the time-series data 
dat = np.genfromtxt(FLIST[0]+'.apx')
snum, xpix, ypix = dat[:,0], dat[:,1], dat[:,2]
NSTARS = len(snum)
# LOOP for marking the comparisons
for j, cidx in enumerate(CLIST1):
    cidx = cidx - 1
    ax.plot(xpix[cidx], ypix[cidx], 'bo', alpha=0.3, ms=15)
    ax.text(xpix[cidx]+30, ypix[cidx], 'C%i' % (j+1,), \
             fontweight='bold', fontsize=15, color='#2222FF')
    ax.text(xpix[cidx]+30, ypix[cidx]+50, '(%i)' % (snum[cidx],), \
             fontweight='bold', fontsize=12, color='#3333FF')
# MARK the target
ax.plot(xpix[TNUM], ypix[TNUM], 'yo', alpha=0.4, ms=20)
ax.text(xpix[TNUM]-30, ypix[TNUM]-70, TARGET, \
        fontweight='bold', fontsize=15, color='r')
ax.text(xpix[TNUM]+30, ypix[TNUM]-20, '(%i)' % (TNUM+1,), \
        fontweight='bold', fontsize=12, color='r')
ax.set_xlim(0,nx)
ax.set_ylim(ny,0)
ax.set_title('Target & Comparions: %s-%s / R_AP=%.1f' % (WNAME,FILTER,PHOT_APER[T_APER-1]))
fig.savefig('w%s-%s-mark' % (WNAME,FILTER))
plt.close('all')

#prnlog('Plot the image with target & comparisons')

# READ the apt files
FLX, ERR, MAG, MRR, FLG = [], [], [], [], []
for i, fidx in enumerate(FLIST):
    dat = np.genfromtxt(fidx+'.apx')
    FLX.append(dat[:,(T_APER+2)])
    ERR.append(dat[:,(N_APER+T_APER+2)])
    MAG.append(dat[:,(2*N_APER+T_APER+2)])
    MRR.append(dat[:,(3*N_APER+T_APER+2)])
    FLG.append(dat[:,(4*N_APER+4)])
    prnlog('FRAME# %4i: Read %s.apx file.' % (FRM[i],fidx))
# MAKE the arrays of photometric results by JDs and stars
# row: frames(time), col: stars(mag, flux) 
FLX, ERR = np.array(FLX), np.array(ERR)
MAG, MRR = np.array(MAG), np.array(MRR)
FLG = np.array(FLG)

# TARGET arrays
TFLX, TERR, TMAG, TMRR, TFLG = \
  FLX[:,TNUM], ERR[:,TNUM], MAG[:,TNUM], MRR[:,TNUM], FLG[:,TNUM]

JD0 = int(JD[0])
# LOOP for stars ==========================================
for cidx1 in CLIST1:
    # CONVERT number into index
    cidx = cidx1 - 1 
    prnlog('%s / MAG PLOT: %03i-%03i' % (WNAME, TNUM1, cidx1))
    # PLOT the light curves of magnitude for each star  
    fig, ax = plt.subplots(num=2, figsize=(10,5))
    y = TMAG - MAG[:,cidx]
    yerr = np.sqrt(TMRR**2 + MRR[:,cidx]**2)
    vv, = np.where(TFLG+FLG[:,cidx] == 0)
    ax.errorbar(JD[vv]-JD0,y[vv],yerr=yerr[vv],fmt='ko',ms=5, alpha=0.5)
    ax.set_ylim(np.max(y[vv])+0.005,np.min(y[vv])-0.005)
    ax.set_title('%s Light Curve (%03d-%03d) JD0=%d' % (WNAME, TNUM1, cidx1, JD0))
    ax.grid()
    fig.savefig('w%s-%s-LC-MAG-%03d' % (WNAME,FILTER,cidx1))
    fig.clf()
    
    # WRITE the magintude difference for each comparison
    fmag = open('w%s-%s-LC-MAG-%03d.txt' % (WNAME,FILTER,cidx1,),'w')
    for vidx in vv:
        fmag.write('%16.8f %8.5f %8.5f \n' % (JD[vidx], y[vidx], yerr[vidx]))
    fmag.close()
plt.close('all')

# CALC. the flux ratio between target and comparisons    
cc = np.array(CLIST1) - 1 

CFLX = np.sum(FLX[:,cc], axis=1)
CERR = np.sqrt(np.sum(ERR[:,cc]**2, axis=1))

SFLX = TFLX / CFLX
SERR = np.sqrt(((TFLX/CFLX**2)**2)*(CERR**2) + ((1.0/CFLX)**2)*(TERR**2))

SFLG = TFLG + np.sum(FLG[:,cc], axis=1)
vv, = np.where(SFLG == 0)

# PLOT the light curve by flux 
fig, ax = plt.subplots(num=3, figsize=(10,5))
ax.errorbar(JD[vv]-JD0, SFLX[vv], yerr=SERR[vv], \
             fmt='ko', ms=4, mew=1, alpha=0.8)
ax.set_title('%s Light Curve (FLX-ALL) JD0=%d' % (WNAME, JD0))
ax.grid()
ax.set_xlabel('JD-JD0')
ax.set_ylabel('Relative Flux')
fig.savefig('w%s-%s-LC-FLX' % (WNAME,FILTER))
fig.clf()
    
# WRITE the light curve of flux ratio 
fout = open('w%s-%s.dat' % (WNAME,FILTER),'w')
for vidx in vv:
    fout.write('%15.6f %12.8f %12.8f \n' % (JD[vidx], SFLX[vidx], SERR[vidx]))
fout.close()

NCOMP = len(CLIST1)
H = max([6,NCOMP*(NCOMP-1)/2])
fig, ax = plt.subplots(num=4, figsize=(8,H))
if NCOMP > 1:
    M0, dM0 = 0.0, 0.03
    for i in range(NCOMP-1,-1,-1):
        for j in range(i-1,-1,-1):
            c1 = CLIST1[i]-1
            c2 = CLIST1[j]-1
            chkG = FLG[:,c1] + FLG[:,c2]
            vv, = np.where(chkG == 0)
            
            chkE = np.sqrt(MRR[:,c1]**2+MRR[:,c2]**2)
            chkM = MAG[:,c1]-MAG[:,c2]
            m = np.median(chkM[vv])
            chkM = chkM[vv] - m 
            chkE = chkE[vv]
            chkT = JD[vv] - JD0
            RMS = np.std(chkM)
            ax.errorbar(chkT, chkM+M0, yerr=chkE, \
                         fmt='o', ms=4, alpha=0.75)
            clabel='C%03i-C%03i\n(%.2f mmag)' % (c1+1,c2+1,RMS*1000) 
            ax.text(np.max(chkT)+0.005,M0-0.2*dM0, clabel)    
            ax.plot(chkT,np.zeros_like(chkT)+M0,'k--',alpha=0.7)
            M0 = M0+dM0
    ax.grid()
    ax.set_ylim(-dM0/2, M0-dM0/2)
    x1, x2 = np.min(chkT),np.max(chkT)
    ax.set_xlim(x1, x2+(x2-x1)*0.25)
    ax.set_ylabel('$\Delta$m')
    ax.set_xlabel('Time [days]')
    ax.set_title('%s $\Delta$m btw. comparisons' % WNAME)
    fig.savefig('w%s-%s-COMPS' % (WNAME,FILTER))
    fig.clf()
plt.close('all')

