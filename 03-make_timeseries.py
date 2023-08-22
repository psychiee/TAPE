# -*- coding: utf-8 -*-
"""
03-make_timeseries
 (1) READ the photometry files (apw)
 (2) FIND the XY-shift of FoV by matching the bright stars (N=20)
 (3) GENERATE the time-series photometry files (apx)

@author: wskang
@update: 2020/05/19
"""
import os
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 
from photlib import read_params, prnlog, sigma_clip
import shutil

# READ the parameter file
par = read_params()

# PARAMETERS for the list of time-series observation
WORKDIR = par['WORKDIR']
LOGFILE = par['LOGFILE']
SHIFT_PLOT = bool(int(par['SHIFTPLOT']))  # ON/OFF image shift plot
N_APER = len(par['PHOTAPER'].split(',')) # number of aperture in .apw file, photometry result
FWHM_CUT2 = np.array(par['FWHMCUT'].split(','),float)[1]
OBSDATE = par['OBSDATE']
TARGETNAME = par['TARGETNAM']
WNAME = OBSDATE+'-'+TARGETNAME

prnlog(f"#WORK: make_timeseries")
prnlog(f"#WORK DIR: {WORKDIR}")
prnlog(f"#SHIFT PLOT: {SHIFT_PLOT}")
prnlog(f"#LOG FILE: {LOGFILE}")

# MOVE to the working directory =======
CDIR = os.path.abspath(os.path.curdir)
os.chdir(WORKDIR)
#======================================
       
# READ log file and 
# SET the fiducial frame index and position 
# automatically find the first frame 
flog = open(LOGFILE,'r')
lfrm, lname, lJD, lX = [], [], [], [] 
for line in flog:
    tmp = line.split()
    lfrm.append(int(tmp[0]))
    lname.append(tmp[1])
    lJD.append(float(tmp[2]))
    lX.append(float(tmp[4]))
FNUM, FLIST, FJD, FX = np.array(lfrm), np.array(lname), np.array(lJD), np.array(lX)
flog.close()

    
# READ the reference frame index and position
FID = FLIST[0]
dat = np.genfromtxt(FID+'.apw')
tx, ty = dat[:,0], dat[:,1] 
tnum = np.arange(1,len(tx)+1, dtype=int)
tmag = dat[:,(2*N_APER+2)]
# ------------------------------------------
# YOU SHOULD CUSTOMIZE THIS NUMBER OF STARS TO MATCH
# ------------------------------------------
NCUT = 47
# ------------------------------------------
tbb = np.argsort(tmag)[:NCUT]

fmat = open('wmatching.log', 'w')
jds, xs, ys, alts, azis, decs = [FJD[0],], [0,], [0,], [0,], [0,], [0,]
# LOOP for apw files
for i, fidx in enumerate(FLIST):

    # READ the FITS header 
    hdu = fits.open(fidx+'.fits')
    img, hdr = hdu[0].data, hdu[0].header
    EXPTIME = float(hdr.get('EXPTIME'))
    FILTER = hdr.get('FILTER')
    HJD = FJD[i]
    ALT = float(hdr.get('ALT'))
    AZI = float(hdr.get('AZ'))
    try:
        DEC = float(hdr.get('DEC'))
    except:
        DEC = 0
    AIRMASS = FX[i]

    # READ the apw file with number and index of apertures 
    dat = np.genfromtxt(fidx+'.apw')
    if len(dat.shape) < 2:
        continue
    
    # READ all coordiates ------------------------------------------
    xpix, ypix = dat[:,0], dat[:,1] 
    # READ brightness data 
    mag = dat[:,(2*N_APER+2)]

    # FILTERING coordinates of bright stars by magnitude-limits
    vv = np.argsort(mag)[:NCUT]
    xpix2, ypix2 = xpix[vv], ypix[vv]
    
    dx, dy = [], [] 
    for inum, ix, iy in zip(tnum[tbb], tx[tbb], ty[tbb]):
        rsq = (ix - xpix2)**2 + (iy - ypix2)**2 
        mm = np.argmin(rsq)
        dx.append(ix - xpix2[mm])
        dy.append(iy - ypix2[mm])
    # CALC. the median of shifts
    xmed, xsig = np.median(dx), np.std(dx)
    ymed, ysig = np.median(dy), np.std(dy)
    
    # PLOT the shift result in the image 
    if SHIFT_PLOT:
        fig, ax = plt.subplots(num=1, figsize=(8,8), dpi=100)
        ny, nx = img.shape
        iavg, imed, istd = sigma_clip(img)
        z1, z2 = imed, imed+istd*8
        ax.imshow(-img, vmin=-z2, vmax=-z1, cmap='gray')
        ax.scatter(xpix, ypix, 100,facecolors='none', edgecolor='b', alpha=0.7)
        ax.plot(tx-xmed, ty-ymed, 'r+', mew=1, ms=18, alpha=0.7, label=f'dx={xmed:.3f}, dy={ymed:.3f}')
        ax.set_xlim(0,nx)
        ax.set_ylim(ny,0)
        ax.set_title('Shift of Image: '+fidx, fontsize=15)
        ax.legend(loc='upper right', fontsize=15)
        fig.savefig(fidx+'-matching')
        fig.clf()
    
    prnlog(f'{fidx} dx={xmed:.2f}({xsig:.2f}) dy={ymed:.2f}({ysig:.2f}) {len(dx):d}/{len(tbb):d}')
    fmat.write(f'{fidx} {xmed:8.3f} {xsig:6.3f} {ymed:8.3f} {ysig:6.3f} {len(dx):d}\n')
    jds.append(HJD)
    xs.append(xmed)
    ys.append(ymed)
    alts.append(ALT)
    azis.append(AZI)
    decs.append(DEC)

    # SAVE photometry result of the same star on the same row
    fdat = open(fidx+'.apx', 'w')         
    for i, ix, iy in zip(tnum, tx, ty):
        rsq = (ix - xpix - xmed)**2 + (iy - ypix - ymed)**2 
        mm = np.argmin(rsq)
        if rsq[mm] > FWHM_CUT2**2:
            flag = 99
        else:
            flag = 0
        fstr = f'{i:03.0f} {xpix[mm]:10.3f} {ypix[mm]:10.3f} '
        fstr1, fstr2, fstr3, fstr4 = '', '', '', ''
        for k in range(N_APER):
            fstr1 = fstr1 + '%12.3f ' % (dat[mm,(k+2)],)
            fstr2 = fstr2 + '%12.3f ' % (dat[mm,(N_APER+k+2)],)
            fstr3 = fstr3 + '%8.3f ' % (dat[mm,(2*N_APER+k+2)],)
            fstr4 = fstr4 + '%8.3f ' % (dat[mm,(3*N_APER+k+2)],)
        fstr += fstr1 + fstr2 + fstr3 + fstr4 
        fstr += '%12.3f %i \n' % (rsq[mm], flag) 
        fdat.write(fstr)
    fdat.close()            

fmat.close()
plt.close('all')

# PLOT the shift map
fig, ax = plt.subplots(num=3, figsize=(8,8), dpi=100)
ax.plot(xs, ys, 'r-', alpha=0.5)
fout = open(f'w{WNAME}-shift.txt', 'w')
num = range(len(xs))
jd0 = jds[0]
djd = jds[-1] - jd0
prnlog(f'#SHIFT PLOT: JD({jd0:.6f} - {djd:.6f})')
for p1, p2, p3, p4, p5, p6, p7 in zip(num, xs, ys, jds, alts, azis, decs):
    fout.write(f'{p1:4.0f} {p4:20.8f} {p2:8.3f} {p3:8.3f} {p5:8.3f} {p6:8.3f} {p7:8.3f}\n')
    ax.plot(p2, p3, 'ro', ms=((p4-jd0)*(3600*12)/500.), alpha=0.5) 
fout.close()
for i in np.linspace(0,djd*3600*12,6):
    ax.plot(-200,200,'ro',ms=(i/500.), alpha=0.5, label=f'{i:5.1f} s')
ax.set_xlim(min(xs)-5,max(xs)+5)
ax.set_ylim(min(ys)-5,max(ys)+5)
ax.set_aspect(1)
ax.grid()
ax.legend(fontsize=10,loc='upper right', ncol=3, numpoints=1)
ax.set_title(f'w{WNAME} Shift Plot')
prnlog(f'WRITE TO w{WNAME}-shift.png...')
fig.savefig(f'w{WNAME}-shift.png')
plt.close('all')

# READ AND PLOT the finding-chart
hdu = fits.open(FID+'.fits')
img, hdr = hdu[0].data, hdu[0].header
ny, nx = img.shape
fig, ax = plt.subplots(num=2, figsize=(9,9), dpi=100)
iavg, imed, istd = sigma_clip(img)
z1, z2 = imed, imed+istd*8
ax.imshow(-img, vmin=-z2, vmax=-z1, cmap='gray')
tpp = np.argsort(tmag)[:200]
# LOOP for the bright stars 
for ix, iy, inum in zip(tx[tpp], ty[tpp], tnum[tpp]):
    ax.text(ix+30, iy, '%02i' % (inum,), fontsize=10, color='#0022FF')
ax.set_xlim(0,nx)
ax.set_ylim(ny,0)
ax.set_title(f'Star Index Chart: {WNAME}-{FILTER} / {FID}')
prnlog(f'WRITE TO w{WNAME}-chart.png...')
fig.savefig(f'w{WNAME}-chart.png')
plt.close('all')

# RETURN to the directory ===========
os.chdir(CDIR) 
#====================================



