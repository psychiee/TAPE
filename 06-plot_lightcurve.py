# -*- coding: utf-8 -*-
"""
05-plot_lightcurve
 (1) READ the time-series data files (apx)
 (2) GENERATE the LC plots with the comparisons
 (3) PLOT the compare btw. comparisons 
 (4) GENERATE the LC data files 

@author: wskang
@update: 2019/09/25
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from photlib import read_params

par = read_params()

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
print ('#WORK DIR: ', par['WORKDIR'])
print ('#TARGET: ', TARGET)
print ('#OBSDATE: ', OBSDATE)
print ('#TARGET INDEX: ', TNUM1)
print ('#COMPARISON INDICES: ', CLIST1)
print ('#LOGFILE: ', LOGFILE)

# CONVERT number into index
TNUM = TNUM1 - 1

# READ the time-series log file 
flog = open(LOGFILE,'r')
lfrm, lname, lJD = [], [], [] 
for line in flog:
    tmp = line.split()
    lfrm.append(int(tmp[0]))
    lname.append(tmp[1])
    lJD.append(float(tmp[2]))
FRM, FLIST, JD = np.array(lfrm), np.array(lname), np.array(lJD)
flog.close()

# SHOW the first image 
try: hdu = fits.open(FLIST[0]+'.fits')
except: hdu = fits.open(FLIST[0]+'.fit') 
FILTER = hdu[0].header.get('FILTER')
img = hdu[0].data
ny, nx = img.shape
fig, ax = plt.subplots(figsize=(9,9))
limg = np.arcsinh(img)
z1, z2 = np.percentile(limg,30), np.percentile(limg,99.5)
ax.imshow(-limg, vmin=-z2, vmax=-z1, cmap='gray')

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
fig.savefig(WNAME+'-'+FILTER+'-mark')
fig.clf()   
print ('plot the image with target & comparisons')

# READ the apt files
FLX, ERR, MAG, MRR, FLG = [], [], [], [], []
for i, fidx in enumerate(FLIST):
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

JD0 = int(JD[0])
# LOOP for stars ==========================================
for cidx1 in CLIST1:
    # CONVERT number into index
    cidx = cidx1 - 1 
    print ('- MAG PLOT: %03i Star' % (cidx1,))
    # PLOT the light curves of magnitude for each star  
    fig1, ax1 = plt.subplots(num=98, figsize=(10,5))
    y = TMAG - MAG[:,cidx]
    yerr = np.sqrt(TMRR**2 + MRR[:,cidx]**2)
    vv, = np.where(TFLG+FLG[:,cidx] == 0)
    ax1.errorbar(JD[vv]-JD0,y[vv],yerr=yerr[vv],fmt='ko',ms=5, alpha=0.5)
    ax1.set_ylim(np.max(y[vv])+0.005,np.min(y[vv])-0.005)
    ax1.set_title(WNAME+' Light Curve (MAG-%03d) JD0=%d' % (cidx1,JD0))
    ax1.grid()
    fig1.savefig('w'+WNAME+'-'+FILTER+'-LC-MAG-%03d' % (cidx1,))
    fig1.clf()
    
    # WRITE the magintude difference for each comparison
    fmag = open('w'+WNAME+'-'+FILTER+'-LC-MAG-%03d.txt' % (cidx1,),'w')
    for vidx in vv:
        fmag.write('%16.8f %8.5f %8.5f \n' % (JD[vidx], y[vidx], yerr[vidx]))
    fmag.close()

# CALC. the flux ratio between target and comparisons    
cc = np.array(CLIST1) - 1 

CFLX = np.sum(FLX[:,cc], axis=1)
CERR = np.sqrt(np.sum(ERR[:,cc]**2, axis=1))

SFLX = TFLX / CFLX
SERR = np.sqrt(((TFLX/CFLX**2)**2)*(CERR**2) + ((1.0/CFLX)**2)*(TERR**2))

SFLG = TFLG + np.sum(FLG[:,cc], axis=1)
vv, = np.where(SFLG == 0)

# PLOT the light curve by flux 
fig3, ax3 = plt.subplots(num=97, figsize=(10,5))
ax3.errorbar(JD[vv]-JD0, SFLX[vv], yerr=SERR[vv], \
             fmt='ko', ms=4, mew=1, alpha=0.8)
ax3.set_title(WNAME+' Light Curve (FLX-ALL) JD0=%d' % (JD0,))
ax3.grid()
ax3.set_xlabel('JD-JD0')
ax3.set_ylabel('Relative Flux')
fig3.savefig('w'+WNAME+'-'+FILTER+'-LC-FLX')
fig3.clf()
    
# WRITE the light curve of flux ratio 
fout = open('w'+WNAME+'-'+FILTER+'.dat','w')
for vidx in vv:
    fout.write('%15.6f %12.8f %12.8f \n' % (JD[vidx], SFLX[vidx], SERR[vidx]))
fout.close()

NCOMP = len(CLIST1)
H = max([6,NCOMP*(NCOMP-1)/2])
fig4, ax4 = plt.subplots(num=96, figsize=(8,H))
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
            ax4.errorbar(chkT, chkM+M0, yerr=chkE, \
                         fmt='o', ms=4, alpha=0.75)
            clabel='C%03i-C%03i\n(%.2f mmag)' % (c1+1,c2+1,RMS*1000) 
            ax4.text(np.max(chkT)+0.005,M0-0.2*dM0, clabel)    
            ax4.plot(chkT,np.zeros_like(chkT)+M0,'k--',alpha=0.7)
            M0 = M0+dM0
    ax4.grid()
    ax4.set_ylim(-dM0/2, M0-dM0/2)
    x1, x2 = np.min(chkT),np.max(chkT)
    ax4.set_xlim(x1, x2+(x2-x1)*0.25)
    ax4.set_ylabel('$\Delta$m')
    ax4.set_xlabel('Time [days]')
    ax4.set_title(WNAME+'; $\Delta$m btw. comparisons')
    fig4.savefig('w'+WNAME+'-'+FILTER+'-COMPS')
    fig4.clf()
plt.close('all')

