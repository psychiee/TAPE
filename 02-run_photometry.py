# -*- coding: utf-8 -*-
"""
02-run_photometry 
 (1) FIND stars in the image (aspylib.astro.find_stars)
 (2) DO aperture photometry 

@author: wskang
@update: 2020/05/19
"""
import sys, os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from astropy.io import fits
from photlib import read_params, helio_jd, calc_jd, find_stars_th, sigma_clip, prnlog, airmass, run_apphot #fit_gauss_elliptical, cal_magnitude
import multiprocessing as mp
import shutil

# READ the parameter file
par = read_params()

# MOVE to the working directory
WORKDIR = par['WORKDIR']
shutil.copy('tape.par', WORKDIR)

os.chdir(WORKDIR)

# ==================================================================
# PARAMETERS for aperture photometry
# ==================================================================
# CCD/Optics parameters
BINNING = int(par['BINNING'])
PSCALE = float(par['PSCALE'])
EGAIN = float(par['EGAIN'])
# Photometry parameters
BOX = int(par['STARBOX'])  # Box size for photometry and centroid
PHOT_APER = np.array(par['PHOTAPER'].split(','), float)  # Radius of aperture
N_APER = len(PHOT_APER)
SKY_ANNUL1, SKY_ANNUL2 = np.array(par['SKYANNUL'].split(','), float)  # inner and outer radius of sky region
SUBPIXEL = int(par['SUBPIXEL'])  # number of subpixel for detailed aperture photometry
# (FINDSTARS) parameters
THRES = float(par['THRES'])  # threshold for detection
SATU = int(par['SATU'])  # saturation level for exclusion
# (CHECK) paramters
FWHM_CUT1, FWHM_CUT2 = np.array(par['FWHMCUT'].split(','), float)
FWHM_ECC = 0.3
# Determine the plots for each star (Warning! too many pngs)
PLOT_FLAG = bool(int(par['STARPLOT']))
# GEN. the log file for the following process
LOGFILE = par['LOGFILE']
# Observatory information for HJD
LAT, LON, H = float(par['OBSLAT']), float(par['OBSLON']), float(par['OBSELEV'])

# -------------------------------------------------------------------------
# GEN. the list of FITS files for aperture photometry
# YOU SHOULD CUSTOMIZE THIS SECTION FOR YOUR FILE NAMING RULES
# ------------------------------------------------------------------------
if len(sys.argv) > 1:
    flist = sys.argv[1:]
else:
    flist = []
    if os.path.exists('wobj.list'):
        dat = np.genfromtxt('wobj.list', dtype='U').flatten()
        for i in range(len(dat)):
            flist.append('w' + dat[i])
    else:
        flist += glob('wobj*.fits')
        flist += glob('?fdz*.fits')
flist.sort()

# DISPLAY the information of photometry
NFRAME = len(flist)
prnlog('#WORK: run_photometry')
prnlog('#WORK DIR: {}'.format(WORKDIR))
prnlog('#TOTAL NUMBER of FILES: {}'.format(NFRAME))
prnlog('#THRES = {}'.format(THRES))
prnlog('#SATU. LEVEL = {}'.format(SATU))
prnlog('#PHOT_APER: {}'.format(PHOT_APER))
prnlog('#SKY_ANNUL: {}'.format([SKY_ANNUL1, SKY_ANNUL2]))
prnlog('#SUBPIXEL: {}'.format(SUBPIXEL))

# SET the file name of observation log
flog = open(LOGFILE, 'w')
# ==================================================================
#  LOOP of images for aperture photometry
# ==================================================================
for i, fname in enumerate(flist):
    fidx = os.path.splitext(fname)[0]

    # READ the FITS file
    hdu = fits.open(fname)[0]
    img, hdr = hdu.data, hdu.header
    ny, nx = img.shape
    DATEOBS = hdr['DATE-OBS']
    TARGET = hdr['OBJECT']
    EXPTIME = hdr['EXPTIME']
    FILTER = hdr['FILTER']
    HJD = hdr.get('HJD')
    if HJD is None:
        try:
            RA = hdr['RA']
            Dec = hdr['Dec']
            HJD = helio_jd(DATEOBS, RA, Dec, exptime=EXPTIME, LAT=LAT, LON=LON, H=H)
        except:
            HJD = calc_jd(DATEOBS, exptime=EXPTIME)

    AIRMASS = hdr.get('AIRMASS')
    if AIRMASS is None:
        try:
            AIRMASS = airmass(hdr.get('ALT'))
        except:
            AIRMASS = 0
    # DISPLAY
    prnlog('#RUN:  %i / %i ' % (i + 1, NFRAME))
    prnlog('#IMAGE/DATE-OBS: %s [%i,%i] %s' % (fidx, nx, ny, DATEOBS))
    prnlog('#OBJECT/EXPTIME/FILTER: {} {} {}'.format(TARGET, EXPTIME, FILTER))

    # IMAGE processing for photometry and plot
    # CALC. the sigma of image
    iavg, imed, istd = sigma_clip(img[BOX:-BOX, BOX:-BOX])
    prnlog(f'#IMAGE STATS: {imed:.2f} ({istd:.2f})')

    # ============================================================================
    # FIND the stars
    # ===========================================================================
    ally, allx = find_stars_th(img, imed + istd * THRES, saturation=(True, SATU),
                               detection_area=int(FWHM_CUT1), margin=BOX)

    # PLOT the finding-chart of frame
    fig, ax = plt.subplots(num=1, figsize=(8, 8), dpi=100)
    z1, z2 = imed, imed + istd * 8
    ax.imshow(-img, vmin=-z2, vmax=-z1, cmap='gray')
    ax.scatter(allx, ally, s=10, facecolors='none', edgecolors='r')
    ax.set_title('Photometry Results: ' + fidx, fontsize=15)
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)

    mp_par = par.copy()
    mp_par.update({'EXPTIME': EXPTIME})
    mp_par.update({'FIDX': fidx})
    mp_par.update({'ISTD': istd})
    mp_par.update({'IMG': img})
    results = []
    for cx, cy in zip(allx, ally):
        results.append(run_apphot(cx, cy, mp_par))
    '''
    pool = mp.Pool(int(mp.cpu_count()/2))
    results = [pool.apply(run_apphot, (cx, cy, mp_par)) for cx, cy in zip(allx, ally)]
    pool.close()
    '''

    # OPEN the output file of Aperture Photometry
    fout = open(fidx + '.apw', 'w')
    # for checking the duplicate stars
    xlist, ylist = [], []
    for fstr in results:
        if fstr is None: continue
        apinfo = fstr.split()
        fx, fy = float(apinfo[0]), float(apinfo[1])
        if len(xlist) >= 1:
            c_rsq = (np.array(xlist) - fx) ** 2 + (np.array(ylist) - fy) ** 2
            if min(c_rsq) < FWHM_CUT2 ** 2:
                prnlog(f'{fidx} [FAIL] {fx:5.0f} {fy:5.0f} rsq = {min(c_rsq):.2f}; TOO CLOSE ')
                continue

        fout.write(fstr + '\n')

        # MARK the star that was completed for photometry
        for Rap in PHOT_APER:
            aper = Circle((fx, fy), Rap, fc='none', ec='b', alpha=0.5, lw=1)
            ax.add_patch(aper)
        # save the coordinate of valid stars
        xlist.append(fx)
        ylist.append(fy)

    # SAVE/CLOSE image figure 
    fig.savefig(fidx + '-phots')
    plt.close('all')

    # CLOSE .apw file
    fout.close()
    prnlog('%i stars of %i completed...' % (len(xlist), len(allx)))

    # WRITE observation log 
    flog.write(f'{i:04.0f} {fidx} {HJD:.8f} {EXPTIME:5.0f} {AIRMASS:9.7f} {FILTER}\n')

# CLOSE log and print files
flog.close()

