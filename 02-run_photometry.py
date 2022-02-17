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
from astropy.modeling import models, fitting
from photlib import read_params, helio_jd, calc_jd, find_stars_th, sigma_clip, prnlog, airmass, fit_gauss_elliptical, cal_magnitude
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
prnlog(f"#WORK: run_photometry")
prnlog(f"#WORK DIR: {WORKDIR}")
prnlog(f"#TOTAL NUMBER of FILES: {NFRAME}")
prnlog(f"#THRES = {THRES}")
prnlog(f"#SATU. LEVEL = {SATU}")
prnlog(f"#PHOT_APER: {PHOT_APER}")
prnlog(f"#SKY_ANNUL: {[SKY_ANNUL1, SKY_ANNUL2]}")
prnlog(f"#SUBPIXEL: {SUBPIXEL}")

# SET the file name of observation log
flog = open(LOGFILE, 'w')

#  LOOP of images for aperture photometry
for inum, fname in enumerate(flist):
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
    prnlog(f"#RUN:  {inum+1} / {NFRAME}")
    prnlog(f"#IMAGE/DATE-OBS: {fidx} {[nx, ny]} {DATEOBS}")
    prnlog(f"#OBJECT/EXPTIME/FILTER: {TARGET} {EXPTIME} {FILTER}")

    # IMAGE processing for photometry and plot
    # CALC. the sigma of image
    iavg, imed, istd = sigma_clip(img[BOX:-BOX, BOX:-BOX])
    prnlog(f"#IMAGE STATS: {imed:.2f} ({istd:.2f})")

    # FIND the stars
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

    results = []
    # LOOP of the stars found by thresholds
    for fx, fy in zip(allx, ally):
        # CHECK the margin stars
        if (fx > nx-BOX) | (fx < BOX) | (fy > ny-BOX) | (fy < BOX): continue

        # DEFINE the box for fitting
        x0, y0 = int(fx) - BOX, int(fy) - BOX
        data = img[(y0):(y0 + 2 * BOX), (x0):(x0 + 2 * BOX)].copy()



        # FIT the stellar profile with 2D Gaussian function
        results.append(fit_gauss_elliptical([y0, x0], data))
        del data

    # OPEN the output file of Aperture Photometry
    fout = open(fidx + '.apw', 'w')
    # for checking the duplicate stars
    xlist, ylist = [], []
    # LOOP of the results by fitting the stars
    for maxi, background, peak, cy, cx, fwhm1, fwhm2, ang in results:
        # CHECK the margin stars
        if (cx > nx - BOX) | (cx < BOX) | (cy > ny - BOX) | (cy < BOX): continue

        # CALC FWHM in arcsec
        pfwhm1, pfwhm2 = fwhm1 * PSCALE, fwhm2 * PSCALE

        # CHECK the condition of star
        error_msg = ''
        if peak < istd:
            error_msg += 'TOO WEAK, '
        if (fwhm2 < FWHM_CUT1) | (fwhm2 > FWHM_CUT2):
            error_msg += 'TOO LARGE/SMALL, '
        if fwhm1 / fwhm2 < FWHM_ECC:
            error_msg += 'TOO OVAL, '
        if len(xlist) >= 1:
            c_rsq = (np.array(xlist) - cx)**2 + (np.array(ylist) - cy)**2
            if min(c_rsq) < FWHM_CUT2 ** 2:
                error_msg += 'TOO CLOSE, '
        if error_msg != '':
            prnlog(f'{fidx} {cx:.0f} {cy:.0f} [FAIL] FWHM={pfwhm2:.1f}" PEAK={peak:.1f} A/B={fwhm1/fwhm2:.1f} ' +
                   f'{error_msg}')
            continue

        # DEFINE the starting index of BOX in the imgae
        x0, y0 = int(cx) - BOX, int(cy) - BOX
        # CROP the image in the BOX
        data = img[(y0):(y0 + 2 * BOX), (x0):(x0 + 2 * BOX)].copy()
        # DEFINE the coordinate of the center in the image
        cx_data, cy_data = (cx - x0), (cy - y0)
        # DEFINE the arrays of coordinates and radii from the center
        yy_data, xx_data = np.indices(data.shape)
        rsq = (xx_data - cx_data)**2 + (yy_data - cy_data)**2

        # DETERMINE sky backgrounds
        cond = (SKY_ANNUL1 ** 2 < rsq) & (rsq < SKY_ANNUL2 ** 2)
        bgdpixels = data[cond].flatten()
        bcnt = len(bgdpixels)
        bavg, bmed, bsig = sigma_clip(bgdpixels)
        bvar = bsig ** 2
        bmed = bmed

        # APPLY subpixel method for precise photometry
        if SUBPIXEL == 1:
            subdata = data
        else:
            subdata = np.zeros([2 * BOX * SUBPIXEL, 2 * BOX * SUBPIXEL])
            for i in range(SUBPIXEL):
                for j in range(SUBPIXEL):
                    subdata[i::SUBPIXEL, j::SUBPIXEL] = data[:, :] / float(SUBPIXEL ** 2)

        # DEFINE the array of radii from the center in sub-pixels
        yy_sub, xx_sub = np.indices(subdata.shape) / float(SUBPIXEL)
        subrsq = (xx_sub - cx_data)**2 + (yy_sub - cy_data)**2

        # DEFINE the list for each aperture
        aflx, aferr, amag, amerr = [], [], [], []
        # LOOP for the aperture sizes
        for k in range(N_APER):
            flxpixels = subdata[subrsq < (PHOT_APER[k]) ** 2].flatten()
            ssum = np.sum(flxpixels)
            scnt = float(len(flxpixels)) / float(SUBPIXEL ** 2)

            # CALC. the total flux and magnitude of the star
            flx, ferr, mag, merr = cal_magnitude(ssum, bmed, bvar, scnt, bcnt, gain=EGAIN)
            mag = mag + 2.5 * np.log10(EXPTIME)
            flx, ferr = flx / EXPTIME, ferr / EXPTIME

            # SAVE into the list
            aflx.append(flx)
            aferr.append(ferr)
            amag.append(mag)
            amerr.append(merr)

        # CHECK the flux
        if min(aflx) < 0:
            error_msg = f'Negative flux={aflx}'
            prnlog(f'{fidx} {cx:5d} {cy:5d} [FAIL] FWHM={pfwhm2:.1f}" PEAK={peak:.1f} A/B={fwhm1/fwhm2:.1f} ' +
                   f'{error_msg}')
            continue

        # WRITE photometry info. into the file
        fstr = '%9.3f %9.3f ' % (cx, cy)
        fstr1, fstr2, fstr3, fstr4 = '', '', '', ''
        for k in range(N_APER):
            fstr1 = fstr1 + '%12.3f ' % (aflx[k],)
            fstr2 = fstr2 + '%12.3f ' % (aferr[k],)
            fstr3 = fstr3 + '%8.4f ' % (amag[k],)
            fstr4 = fstr4 + '%8.4f ' % (amerr[k],)
        fstr = fstr + fstr1 + fstr2 + fstr3 + fstr4
        fstr = fstr + '%12.3f %12.3f' % (bmed, bsig)
        fout.write(fstr + "\n")

        # MARK the star that was completed for photometry
        for Rap in PHOT_APER:
            aper = Circle((cx, cy), Rap, fc='none', ec='b', alpha=0.5, lw=1)
            ax.add_patch(aper)

        # SAVE the coordinate of valid stars
        xlist.append(cx)
        ylist.append(cy)

        # PLOT the each star profile and image
        if PLOT_FLAG:

            figs, (ax0, ax1, ax2) = plt.subplots(num=99, ncols=3, figsize=(12, 3.5))
            ax0.set_aspect(1)
            ax1.set_aspect(1)
            # plot the box image
            cmin, cmax = background, background + peak
            ax0.imshow(-data, vmax=-cmin, vmin=-cmax, cmap='gray')
            # plot the aperture in the box image
            for k in range(len(PHOT_APER)):
                apr = Wedge([cx_data, cy_data], PHOT_APER[k], 0, 360, width=BOX/500, color='r', alpha=0.8)
                ax0.add_patch(apr)
            # plot the sky annulus in the box image
            a_mid = (SKY_ANNUL1 + SKY_ANNUL2) / 2.0
            a_wid = (SKY_ANNUL2 - SKY_ANNUL1) / 2.0
            ann = Wedge([cx_data, cy_data], a_mid, 0, 360, width=a_wid, color='g', alpha=0.3)
            ax0.add_patch(ann)
            ax0.plot(cx_data, cy_data, 'r+', ms=10, mew=1)
            ax0.set_xlim(0, 2 * BOX)
            ax0.set_ylim(2 * BOX, 0)

            # plot the contour
            levels = np.linspace(cmin, cmax, 10)
            y, x = np.indices(data.shape)
            ax1.contour(x, y, data, levels[2:])
            ax1.set_ylim(2 * BOX - 0.5, -0.5)
            ax1.set_xlim(-0.5, 2 * BOX - 0.5)
            # plot the semi-major axis in contour
            dx, dy = fwhm2 / 2 * np.sin(ang), fwhm2 / 2 * np.cos(ang)
            ax1.plot([cx_data - dx, cx_data + dx], [cy_data - dy, cy_data + dy], 'r-', lw=4, alpha=0.5)
            # plot the semi-minor axis in contour
            dx, dy = fwhm1 / 2 * np.sin((ang - np.pi / 2)), fwhm1 / 2 * np.cos((ang - np.pi / 2))
            ax1.plot([cx_data - dx, cx_data + dx], [cy_data - dy, cy_data + dy], 'b-', lw=4, alpha=0.5)
            ax1.set_title(f'(X,Y)=({cx:.0f},{cy:.0f})')
            ax1.grid()

            # plot the radial profile
            rsq_data = (xx_data - cx_data)**2 + (yy_data - cy_data)**2
            rad_1d = np.sqrt(rsq_data.flatten())
            pix_1d = data.flatten() - background
            ss = np.argsort(rad_1d)
            rad_x, pix_y = rad_1d[ss], pix_1d[ss]
            GM1 = models.Gaussian1D(amplitude=peak, mean=0, stddev=0.5 * fwhm1 / np.sqrt(2 * np.log(2)))
            GM2 = models.Gaussian1D(amplitude=peak, mean=0, stddev=0.5 * fwhm2 / np.sqrt(2 * np.log(2)))
            ax2.plot(rad_x, pix_y, 'k.', ms=3, alpha=0.2)
            ax2.plot(rad_x, GM1(rad_x), 'b-', alpha=0.8, label=f'FWHM1 {fwhm1:5.1f} {pfwhm1:5.2f}')
            ax2.plot(rad_x, GM2(rad_x), 'r-', alpha=0.8, label=f'FWHM2 {fwhm2:5.1f} {pfwhm2:5.2f}')
            ax2.set_xlabel('Radius [pix]')
            ax2.set_ylabel('Pix. Value')
            ax2.set_xlim(0,fwhm2*3)
            ax2.legend(loc='upper right')
            ax2.set_title(f'(BGD,PEAK)=({background:.0f},{peak:.0f})')
            #ax2.text(0.5, 0.99, strtmp, fontweight='bold', color='k', alpha=0.7, ha='center', va='top', transform=ax2.transAxes, fontsize=12)
            ax2.grid()

            if not os.path.exists('./prof/'):
                os.makedirs('./prof/')
            figs.savefig(f'./prof/{fidx}-{cx:04.0f}_{cy:04.0f}.png')
            figs.clf()
            del data, subdata, rsq, subrsq

    # SAVE/CLOSE image figure 
    fig.savefig(fidx + '-phots')
    plt.close('all')

    # CLOSE .apw file
    fout.close()
    prnlog('%i stars of %i completed...' % (len(xlist), len(allx)))

    # WRITE observation log 
    flog.write(f'{inum:04.0f} {fidx} {HJD:.8f} {EXPTIME:5.0f} {AIRMASS:9.7f} {FILTER}\n')

# CLOSE log and print files
flog.close()