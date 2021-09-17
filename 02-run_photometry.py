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
from photlib import read_params, helio_jd, calc_jd, find_stars_th, \
    fit_gauss_elliptical, cal_magnitude, sigma_clip, prnlog, airmass

# READ the parameter file
par = read_params()

# MOVE to the working directory
WORKDIR = par['WORKDIR']
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
    try:
        GAIN = hdr['EGAIN']
    except:
        GAIN = EGAIN
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

    # OPEN the output file of Aperture Photometry 
    fout = open(fidx + '.apw', 'w')
    # MAKE the lists for checking duplicate stars. 
    xlist, ylist = [], []
    # SORT the stars using X-axis position
    # ss = np.argsort(ax)
    # LOOP for each star
    for cx, cy in zip(allx, ally):

        # EXCLUDE the stars near the border 
        if (cx < BOX) | (cy < BOX) | (cx > nx - BOX) | (cy > ny - BOX):
            continue

        # DEFINE the box for photometry (cropping) 
        x0, y0 = int(cx) - BOX, int(cy) - BOX
        data = img[(y0):(y0 + 2 * BOX), (x0):(x0 + 2 * BOX)].copy()

        # FIT the star profile with 2D Moffat function 
        maxi, background, peak, y_cen, x_cen, fwhm1, fwhm2, ang = \
            fit_gauss_elliptical([y0, x0], data)
        pfwhm = fwhm2 * PSCALE
        # PRINT stellar statistics

        # CHECK the condition of star
        error_msg = ''
        if len(xlist) > 1:
            crsq = (np.array(xlist) - x_cen) ** 2 + (np.array(ylist) - y_cen) ** 2
            if min(crsq) < (FWHM_CUT2) ** 2:
                error_msg += 'TOO CLOSE, '
        if peak < istd:
            error_msg += 'TOO WEAK, '
        if (fwhm2 < FWHM_CUT1) | (fwhm2 > FWHM_CUT2):
            error_msg += 'TOO LARGE/SMALL, '
        if fwhm1 / fwhm2 < FWHM_ECC:
            error_msg += 'TOO OVAL, '
        if (x_cen - cx) ** 2 + (y_cen - cy) ** 2 > FWHM_CUT2 ** 2:
            error_msg += 'FIT-OFFSET, '

        if error_msg != '':
            prnlog(f'{fidx} {cx:5d} {cy:5d} [FAIL] FWHM=[{fwhm1:.1f},{fwhm2:.1f}] PEAK={peak:.1f}  {error_msg}')
            continue

        # ----------------------------------------------------------------
        # APPLY subpixel method for precise photometry
        # ----------------------------------------------------------------
        imgy, imgx = np.indices(data.shape)
        imgy, imgx = imgy + y0, imgx + x0
        rsq = (imgx - x_cen) ** 2 + (imgy - y_cen) ** 2
        if SUBPIXEL == 1:
            subdata = data.copy()
            rsq_sub = rsq.copy()
        else:
            subdata = np.zeros([2 * BOX * SUBPIXEL, 2 * BOX * SUBPIXEL])
            for i in range(SUBPIXEL):
                for j in range(SUBPIXEL):
                    subdata[i::SUBPIXEL, j::SUBPIXEL] = data[:, :] / float(SUBPIXEL ** 2)
            suby, subx = np.indices(subdata.shape) / float(SUBPIXEL)
            subx, suby = subx + x0, suby + y0
            rsq_sub = (subx - x_cen) ** 2 + (suby - y_cen) ** 2

            # ----------------------------------------------------------------
        # DETERMINE sky backgrounds 
        # ----------------------------------------------------------------
        cond = (SKY_ANNUL1 ** 2 < rsq) & (rsq < SKY_ANNUL2 ** 2)
        bgdpixels = data[cond].flatten()
        bcnt = len(bgdpixels)
        bavg, bmed, bsig = sigma_clip(bgdpixels)
        bvar = bsig ** 2

        # DEFINE the list for each aperture 
        aflx, aferr, amag, amerr = [], [], [], []
        # LOOP for the aperture sizes
        for k in range(N_APER):
            flxpixels = subdata[rsq_sub < (PHOT_APER[k]) ** 2].flatten()
            ssum = np.sum(flxpixels)
            scnt = float(len(flxpixels)) / float(SUBPIXEL ** 2)

            # CALC. the total flux and magnitude of the star 
            flx, ferr, mag, merr = cal_magnitude(ssum, bmed, bvar, scnt, bcnt, gain=GAIN)
            mag = mag + 2.5 * np.log10(EXPTIME)
            flx, ferr = flx / EXPTIME, ferr / EXPTIME

            # SAVE into the list 
            aflx.append(flx)
            aferr.append(ferr)
            amag.append(mag)
            amerr.append(merr)

        if min(aflx) < 0:
            error_msg = 'Negative flux={}'.format(aflx)
            prnlog(
                f'{fidx} {cx:5d} {cy:5d} [FAIL] FWHM={pfwhm:.1f}" PEAK={peak:.1f} A/B={fwhm1 / fwhm2:.1f} {error_msg}')
            continue

        # WRITE photometry info. into the file
        fstr = '%9.3f %9.3f ' % (x_cen, y_cen)
        fstr1, fstr2, fstr3, fstr4 = '', '', '', ''
        for k in range(N_APER):
            fstr1 = fstr1 + '%12.3f ' % (aflx[k],)
            fstr2 = fstr2 + '%12.3f ' % (aferr[k],)
            fstr3 = fstr3 + '%8.3f ' % (amag[k],)
            fstr4 = fstr4 + '%8.3f ' % (amerr[k],)
        fstr = fstr + fstr1 + fstr2 + fstr3 + fstr4
        fstr = fstr + '%12.3f %12.3f' % (bmed, bsig)
        fout.write(fstr + '\n')

        # SAVE star-coord. for checking duplicate stars         
        xlist.append(x_cen)
        ylist.append(y_cen)

        # MARK the star that was completed for photometry  
        for Rap in PHOT_APER:
            aper = Circle((x_cen, y_cen), Rap, fc='none', ec='b', alpha=0.5, lw=1)
            ax.add_patch(aper)

        # ==========================================================================
        # PLOT the each star profile and image 
        # ==========================================================================
        if not PLOT_FLAG:
            continue

        figs, (ax0, ax1, ax2) = plt.subplots(num=99, ncols=3, figsize=(12, 3.5))
        ax0.set_aspect(1)
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        # plot the box image 
        cmin, cmax = background, background + peak
        ax0.imshow(-subdata, vmax=(-cmin/SUBPIXEL**2), vmin=(-cmax/SUBPIXEL**2), cmap='gray')
        tt = np.linspace(0, 2 * np.pi, 200)
        # plot the aperture in the box image 
        sub_xc, sub_yc = SUBPIXEL * (x_cen - x0), SUBPIXEL * (y_cen - y0)
        for k in range(len(PHOT_APER)):
            apr = Wedge([sub_xc, sub_yc], SUBPIXEL * PHOT_APER[k], 0, 360, width=BOX / 500, color='r', alpha=0.8)
            ax0.add_patch(apr)
        # plot the sky annulus in the box image                         
        a_mid = SUBPIXEL * (SKY_ANNUL1 + SKY_ANNUL2) / 2.0
        a_wid = SUBPIXEL * (SKY_ANNUL2 - SKY_ANNUL1) / 2.0
        ann = Wedge([sub_xc, sub_yc], a_mid, 0, 360, width=a_wid, color='g', alpha=0.3)
        ax0.add_patch(ann)
        # mark the center of star 
        ax0.plot(sub_xc, sub_yc, 'r+', ms=10, mew=1)
        ax0.set_xlim(0, 2 * BOX * SUBPIXEL)
        ax0.set_ylim(2 * BOX * SUBPIXEL, 0)
        # plot the contour 
        levels = np.linspace(cmin, cmax, 8)
        y, x = np.indices(data.shape)
        ax1.contour(x, y, data, levels)
        ax1.set_ylim(2 * BOX - 0.5, -0.5)
        ax1.set_xlim(-0.5, 2 * BOX - 0.5)
        # plot the semi-major axis in contour
        dx, dy = fwhm2 / 2 * np.sin(ang), fwhm2 / 2 * np.cos(ang)
        ax1.plot([x_cen - x0 - dx, x_cen - x0 + dx], [y_cen - y0 - dy, y_cen - y0 + dy], 'r-', lw=4, alpha=0.5)
        # plot the semi-minor axis in contour
        dx, dy = fwhm1 / 2 * np.sin((ang - np.pi / 2)), fwhm1 / 2 * np.cos((ang - np.pi / 2))
        ax1.plot([x_cen - x0 - dx, x_cen - x0 + dx], [y_cen - y0 - dy, y_cen - y0 + dy], 'b-', lw=4, alpha=0.5)
        ax1.set_title('(%04d,%04d)' % (x_cen, y_cen))
        ax1.grid()
        # plot the semi-major/minor axis in the frame
        dx, dy = 10 * fwhm2 * np.sin(ang), 10 * fwhm2 * np.cos(ang)
        ax2.plot([x_cen - dx, x_cen + dx], [y_cen - dy, y_cen + dy], 'r-', lw=3)
        dx, dy = 10 * fwhm1 * np.sin((ang - np.pi / 2)), 10 * fwhm1 * np.cos((ang - np.pi / 2))
        ax2.plot([x_cen - dx, x_cen + dx], [y_cen - dy, y_cen + dy], 'b-', lw=3)
        ax2.set_ylim(ny, 0)
        ax2.set_xlim(0, nx)
        # text the information of fitting 
        strtmp = '(BGD,PEAK)=(%i,%i) \nFWHM=%.3f\" PA=%+id' % \
                 (background, peak, pfwhm, np.rad2deg(ang))
        ax2.text(0.5, 0.99, strtmp, fontweight='bold', color='k', alpha=0.7, ha='center', va='top',
                 transform=ax2.transAxes, fontsize=12)
        ax2.grid()
        figs.savefig(fidx + '-%04d_%04d' % (cx, cy))
        figs.clf()

        del data, subdata, rsq_sub

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
