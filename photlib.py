# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:19:01 2018

@author: wskang
@update: 2020/05/19
"""
import numpy as np 
import scipy.ndimage as sn
import scipy.optimize as so
import scipy.stats as ss
from astropy import time as aptime, coordinates as apcoord
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt 
import time 

#LAT, LON, H = 34.5261362, 127.4470482, 81.35789

def prnlog(text):
    with open('system.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+text+'\n')
    print(text)
    return 
        

def read_params():
    f = open('tape.par', 'r')
    par = {}
    for line in f: 
        tmp = line.split()
        par.update({tmp[0]:tmp[1]})
    return par 


def sigma_clip1(xx, lower=3, upper=3):
    mxx = np.ma.masked_array(xx)
    while 1:
        xavg, xsig = np.mean(mxx), np.std(mxx)
        mask = (mxx < (xavg - lower*xsig)) | (mxx > (xavg + upper*xsig))
        if (mask == mxx.mask).all():
            break
        mxx.mask = mask
    return np.mean(mxx), np.std(mxx)


def sigma_clip2(xx, lower=3, upper=3):
    """
    OUTPUT: average, median, stddev
    """
    return sigma_clipped_stats(xx, sigma_lower=lower, sigma_upper=upper)


def sigma_clip3(xx, lower=3, upper=3):
    """
    OUTPUT: average, median, stddev
    """
    mxx, _, _ = ss.sigmaclip(xx, lower, upper)
    return np.mean(mxx), np.median(mxx), np.std(mxx) 


def sigma_clip(xx, lower=3, upper=3):
    return sigma_clip3(xx)


def helio_jd(dateobs, ra, dec, exptime=0, LAT=0, LON=0, H=0):
    objcoo = apcoord.SkyCoord(ra, dec, unit=('hourangle', 'deg'), frame='fk5')
    doao = apcoord.EarthLocation.from_geodetic(LON, LAT, H)
    times = aptime.Time(dateobs, format='isot', scale='utc', location=doao)
    dt = aptime.TimeDelta(exptime / 2.0, format='sec')
    times = times + dt
    # ltt_bary = times.light_travel_time(objcoo)
    # times_bary = times.tdb + ltt_bary
    ltt_helio = times.light_travel_time(objcoo, 'heliocentric')
    times_helio = times.utc + ltt_helio

    return times_helio.jd


def airmass(alt):
    """Computes the airmass at a given altitude.
    Inputs:
    - alt   the observed altitude, as affected by refraction (deg)
    Returns an estimate of the air mass, in units of that at the zenith.
    Warnings:   
    - Alt < _MinAlt is treated as _MinAlt to avoid arithmetic overflow.
    History:
    Original code by P.W.Hill, St Andrews
    Adapted by P.T.Wallace, Starlink, 5 December 1990
    2002-08-02 ROwen  Converted to Python
    """
    if alt is None: return -1
    
    _MinAlt = 3.0
    # secM1 is secant(zd) - 1 where zd = 90-alt
    secM1 = (1.0 / np.sin(max(_MinAlt, alt)*np.pi/180)) - 1.0
    return 1.0 + secM1 * (0.9981833 - secM1 * (0.002875 + (0.0008083 * secM1)))

######################################################
# FUNCTIONS for aperture photometry 
######################################################


def detect_peaks(image, detection_area=2):
    """
    ---------------------
    Purpose
    Takes an image and detects the peaks using a local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when the pixel's value is a local maximum, 0 otherwise)
    ---------------------
    Inputs
    * image (2D Numpy array) = the data of one FITS image, as obtained for instance by reading one FITS image file with the function astro.get_imagedata().
    * detection_area (integer) = optional argument. The local maxima are found in a square neighborhood with size (2*detection_area+1)*(2*detection_area+1). Default value is detection_area = 2.
    ---------------------
    Output (2D Numpy array) = Numpy array with same size as the input image, with 1 at the image local maxima, 0 otherwise.
    ---------------------
    """
    # define an 8-connected neighborhood --> image dimensionality is 2 + there are 8 neighbors for a single pixel
    neighborhood = np.ones((1+2*detection_area, 1+2*detection_area))
    # apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
    local_max = sn.maximum_filter(image, footprint=neighborhood) == image
    # ocal_max is a mask that contains the peaks we are looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # We create the mask of the background
    background = (image == 0)
    # a little technicality: we must erode the background in order to successfully subtract it from local_max,
    # otherwise a line will appear along the background border (artifact of the local maximum filter)
    eroded_background = sn.binary_erosion(background, structure=neighborhood, border_value=1)
    # we obtain the final mask, containing only peaks, by removing the background from the local_max mask
    # detected_peaks = local_max - eroded_background
    detected_peaks = np.logical_xor(local_max, eroded_background)
    return detected_peaks

    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710     
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html

def detect_stars(image, threshold, detection_area = 2, saturation = [False,0], margin = 5):
    """
    ---------------------
    Purpose
    Takes an image and detects all objects above a certain threshold.
    Returns a boolean mask giving the objects positions (i.e. 1 when the pixel's value is at an object's center, 0 otherwise)
    ---------------------
    Inputs
    * image (2D Numpy array) = the data of one FITS image, as obtained for instance by reading one FITS image file with the function astro.get_imagedata().
    * threshold (integer) = the objects are only detected if their maximum signal is above threshold.
    * detection_area (integer) = optional argument. The object's local maxima are found in a square neighborhood with size (2*detection_area+1)*(2*detection_area+1). Default value is detection_area = 2.
    * saturation (list) = optional argument. Two elements lists: if saturation[0] is True, the objects with maximum signal >= saturation[1] are disregarded.
    * margin (integer or list) = optional argument. The default value is 5. The objects having their center closer than margin pixels from the image border are disregarded. Margin can also be a list of 4 integers, in which case it represents the top, bottom, right, left margins respectively.
    ---------------------
    Output (2D Numpy array) = Numpy array with same size as the input image, with 1 at the objects centers, 0 otherwise.
    ---------------------
    """
    sigmafilter = 1

    # --- threshold ---
    imageT = image >= threshold

    # --- smooth threshold, mask and filter initial image ---
    neighborhood = sn.generate_binary_structure(2,1)
    imageT = sn.binary_erosion(imageT, structure=neighborhood, border_value=1)
    imageT = sn.binary_dilation(imageT, structure=neighborhood, border_value=1)
    imageF = sn.filters.gaussian_filter(imageT*image,sigmafilter)

    # --- find local max ---
    peaks = detect_peaks(imageF, detection_area = detection_area)
    if isinstance(margin,list):
        peaks[-margin[0]:,:]=0
        peaks[:margin[1],:]=0
        peaks[:,-margin[2]:]=0
        peaks[:,:margin[3]]=0
    else:
        peaks[-margin:,:]=0
        peaks[:margin,:]=0
        peaks[:,-margin:]=0
        peaks[:,:margin]=0

    # --- removes too bright stars (possibly saturated) ---
    if saturation[0]:
        peaks = peaks * (image<saturation[1])
    
    return peaks

    
def find_stars(image, nb_stars, excluded=[], included=[], detection_area=2, saturation=[False,0], margin=5, text_display=True):
    """
    ---------------------
    Purpose
    Takes an image and returns the approximate positions of the N brightest stars, N being specified by the user.
    ---------------------
    Inputs
    * image (2D Numpy array) = the data of one FITS image, as obtained for instance by reading one FITS image file with the function astro.get_imagedata().
    * nb_stars (integer) = the number of stars requested.
    * excluded (list) = optional argument. List of 4 elements lists, with the form [[x11,x12,y11,y12], ..., [xn1,xn2,yn1,yn2]]. Each element is a small part of the image that will be excluded for the star detection.
    This optional parameter allows to reject problematic areas in the image (e.g. saturated star(s), artefact(s), moving object(s)..)
    * included (list) = optional argument. List of 4 integers, with the form [x1,x2,y1,y2]. It defines a subpart of the image where all the stars will be detected.
    Both excluded and included can be used simultaneously.
    * detection_area (integer) = optional argument. The object's local maxima are found in a square neighborhood with size (2*detection_area+1)*(2*detection_area+1). Default value is detection_area = 2.
    * saturation (list) = optional argument. Two elements lists: if saturation[0] is True, the objects with maximum signal >= saturation[1] are disregarded.
    * margin (integer) = optional argument. The objects having their center closer than margin pixels from the image border are disregarded.
    * text_display (boolean) = optional argument. If True then some information about the star detection is printed in the console.
    ---------------------
    Output (2D Numpy array) = Numpy array with the form [[x1,y1],[x2,y2],...,[xn,yn]], that contains the approximate x and y positions for each star.
    ---------------------
    """

    if text_display:
        print ("--- detecting stars ---")
        print ("threshold, nb of detected stars:")

    # --- included ---
    if len(included)>0:
        x0, x1, y0, y1 = included
        im = image[x0:x1,y0:y1]
    else:
        im = image
        x0, y0 = [0,0]
    
    # --- get subimage information ---
    maxi = np.max(im.flatten())
    # median = stats.get_median(im)[0]
    medi = np.median(im)
        
    # --- searching threshold ---
    th = medi + 0.1*maxi
    peaks = detect_stars(im, th, detection_area = detection_area, saturation = saturation, margin = margin)
    for area in excluded:
        x2,x3,y2,y3 = area
        peaks[max(x2-x0,0):max(x3-x0,0),max(y2-y0,0):max(y3-y0,0)] = 0
    x, y = np.where(peaks==1)
    x = x + x0
    y = y + y0
    nb = len(x)
    if text_display:
        print (th, nb)

    iter = 0
    while nb<nb_stars:
        iter +=1
        th = th*0.1+medi*0.9
        peaks = detect_stars(im, th, detection_area = detection_area, saturation = saturation, margin = margin)
        for area in excluded:
            x2,x3,y2,y3 = area
            peaks[max(x2-x0,0):max(x3-x0,0),max(y2-y0,0):max(y3-y0,0)] = 0
        x, y = np.where(peaks==1)
        x = x + x0
        y = y + y0
        nb = len(x)
        if text_display:
            print (th, nb)
        if iter>10:
            raise Exception("error in aspylib.astro.find_stars(), star detection does not converge")

    xy = np.vstack([x,y]).transpose()
    maxi = image[list(x),list(y)]
    xy = xy[np.flipud(np.argsort(maxi)),:]

    if text_display:
        print ("number of detected stars =",nb_stars)

    return xy[:nb_stars,:]


def find_stars_th(image, threshold, excluded=[], included=[], detection_area=2, saturation=[False,0], margin=5, text_display=True):
    """
    ---------------------
    Purpose
    Takes an image and returns the approximate positions of the N brightest stars, N being specified by the user.
    ---------------------
    Inputs
    * image (2D Numpy array) = the data of one FITS image, as obtained for instance by reading one FITS image file with the function astro.get_imagedata().
    * nb_stars (integer) = the number of stars requested.
    * threshold (float) = detection limit of pixel value 
    * excluded (list) = optional argument. List of 4 elements lists, with the form [[x11,x12,y11,y12], ..., [xn1,xn2,yn1,yn2]]. Each element is a small part of the image that will be excluded for the star detection.
    This optional parameter allows to reject problematic areas in the image (e.g. saturated star(s), artefact(s), moving object(s)..)
    * included (list) = optional argument. List of 4 integers, with the form [x1,x2,y1,y2]. It defines a subpart of the image where all the stars will be detected.
    Both excluded and included can be used simultaneously.
    * detection_area (integer) = optional argument. The object's local maxima are found in a square neighborhood with size (2*detection_area+1)*(2*detection_area+1). Default value is detection_area = 2.
    * saturation (list) = optional argument. Two elements lists: if saturation[0] is True, the objects with maximum signal >= saturation[1] are disregarded.
    * margin (integer) = optional argument. The objects having their center closer than margin pixels from the image border are disregarded.
    * text_display (boolean) = optional argument. If True then some information about the star detection is printed in the console.
    ---------------------
    Output (2D Numpy array) = Numpy array with the form [[x1,y1],[x2,y2],...,[xn,yn]], that contains the approximate x and y positions for each star.
    ---------------------
    """
    # --- included ---
    if len(included)>0:
        x0, x1, y0, y1 = included
        im = image[x0:x1,y0:y1]
    else:
        im = image
        x0, y0 = [0,0]
    
    # --- searching threshold ---
    peaks = detect_stars(im, threshold, detection_area = detection_area, saturation = saturation, margin = margin)
    for area in excluded:
        x2,x3,y2,y3 = area
        peaks[max(x2-x0,0):max(x3-x0,0),max(y2-y0,0):max(y3-y0,0)] = 0
    x, y = np.where(peaks==1)
    x = x + x0
    y = y + y0
    nb = len(x)
    if text_display:
        prnlog(" - NUMBER of detected stars: %i" % nb)

    return x, y


def fit_gauss_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical gaussian PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 8 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results),
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    ---------------------
    """

    # find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    # if star is saturated it could be that median value is 32767 or 65535 --> height=0
    if height == 0.0:
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm
    sig_1 = fwhm_1 / (2.*np.sqrt(2.*np.log(2.)))
    sig_2 = fwhm_2 / (2.*np.sqrt(2.*np.log(2.)))

    angle = 0.

    p0 = floor, height, mean_x, mean_y, sig_1, sig_2, angle

    # ---------------------------------------------------------------------------------
    # fitting gaussian
    def gauss(floor, height, mean_x, mean_y, sig_1, sig_2, angle):

        A = (np.cos(angle)/sig_1)**2. + (np.sin(angle)/sig_2)**2.
        B = (np.sin(angle)/sig_1)**2. + (np.cos(angle)/sig_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./(sig_1**2.)-1./(sig_2**2.))

        #do not forget factor 0.5 in exp(-0.5*r**2./sig**2.)
        return lambda x,y: floor + height*np.exp(-0.5*(A*((x-mean_x)**2)+B*((y-mean_y)**2)+C*(x-mean_x)*(y-mean_y)))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)

    p = so.leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]

    # ---------------------------------------------------------------------------------
    # formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]

    # angle gives the direction of the p[4]=sig_1 axis, starting from x (vertical) axis,
    # clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        angle = np.arctan(np.tan(p[6]))
    # then sig_1 is the smallest : we want angle to point to sig_y, the largest
    else:

        fwhm_large = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        angle = np.arctan(np.tan(p[6]+np.pi/2.))

    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]
    return output


def fit_moffat_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical moffat PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 9 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    - beta is the "beta" parameter of the moffat function    
    ---------------------
    """
    
    # ---------------------------------------------------------------------------------
    # find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    # if star is saturated it could be that median value is 32767 or 65535 --> height=0
    if height==0.0: 
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm

    angle = 0.0
    beta = 4
    
    p0 = floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta

    # ---------------------------------------------------------------------------------
    # fitting gaussian
    def moffat(floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta):
        beta = abs(beta)
        alpha_1 = 0.5*fwhm_1/np.sqrt(2**(1/beta)-1)
        alpha_2 = 0.5*fwhm_2/np.sqrt(2**(1/beta)-1)
    
        A = (np.cos(angle)/alpha_1)**2. + (np.sin(angle)/alpha_2)**2.
        B = (np.sin(angle)/alpha_1)**2. + (np.cos(angle)/alpha_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./alpha_1**2. - 1./alpha_2**2.)
        
        return lambda x,y: floor + height/((1.+ A*((x-mean_x)**2) + B*((y-mean_y)**2) + C*(x-mean_x)*(y-mean_y))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = so.leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    # ---------------------------------------------------------------------------------
    # formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    beta = p[7]
    
    # angle gives the direction of the p[4]=fwhm_1 axis, starting from x (vertical) axis,
    # clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):
        fwhm_large = np.abs(p[4])
        fwhm_small = np.abs(p[5])
        angle = np.arctan(np.tan(p[6]))
    # then fwhm_1 is the smallest : we want angle to point to sig_y, the largest
    else:
        fwhm_large = np.abs(p[5])
        fwhm_small = np.abs(p[4])
        angle = np.arctan(np.tan(p[6]+np.pi/2.))

    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]
    return output


def cal_magnitude(source, bmed, bvarpix, Ns, Nb, gain=1.0, zeropoint=20.0):
    
    # flux in ADU
    flx = (source - Ns * bmed)
    
    # flux error in photons
    fvar = flx / gain + (Ns + (np.pi/2) * Ns**2 / Nb) * bvarpix
    ferr = np.sqrt(fvar)
    if flx < 0: 
        mag, merr = -99, -99
    else:
        mag = -2.5 * np.log10(flx*gain) + zeropoint 
        merr = (2.5*gain/(flx*np.log(10))) * ferr
    
    return flx, ferr, mag, merr

###################################################################
# HOPS code for distribution of image pixels, finding stars
###################################################################


def dist_gaussian(image, binsize=5.0, cutoff=0.05, PLOT=False):
    
    values = image.flatten()
    minv, maxv = np.min(values), np.max(values)
    avgv, medv, sigv = sigma_clip(values)
    
    nbins = int((maxv - minv) / binsize) + 1
    xbins = np.array(minv + np.arange(nbins)*binsize + binsize/2)
    int_values = np.array((values - minv)/binsize, int)
    ycnts = np.bincount(int_values)
    # from HOPS functions 
    # ycnts = np.insert(ycnts, len(ycnts), np.zeros(int(nbins) - len(ycnts)))
    del values
    
    pidx = np.argmax(ycnts)
    xpeak, ypeak = xbins[pidx], ycnts[pidx]
   
    fit_g = fitting.LevMarLSQFitter()
 
    vv = np.where((xbins < xpeak + 3*sigv) & (xbins > xpeak - 3*sigv))[0]
    g_init = models.Gaussian1D(amplitude=ypeak, mean=xpeak, stddev=sigv)
    g = fit_g(g_init, xbins[vv], ycnts[vv])
    # print(g.mean.value, g.stddev.value)
    
    if PLOT:
        plt.figure()
        plt.plot(xbins[vv], ycnts[vv], 'ko-')
        plt.plot(xbins[vv], g(xbins[vv]), 'r-', lw=4, alpha=0.5, \
                  label='%.1f(%.1f)' % (g.mean.value, g.stddev.value))
        plt.legend()
        plt.savefig('g_dist_%i_%i' % (xpeak, sigv))
        plt.close('all')
    
    return g.mean.value, g.stddev.value


def r_cumulated(image, binsize=5):
    
    values = image.flatten()
    npix = len(values)
    minv, maxv = np.min(values), np.max(values)
    sigv = np.std(values)
    
    nbins = int((maxv - minv) / binsize) + 1
    xbins = np.array(minv + np.arange(nbins)*binsize + binsize/2)
    int_values = np.array((values - minv)/binsize, int)
    ycnts = np.bincount(int_values)    

    pidx = np.argmax(ycnts)
    xpeak, ypeak = xbins[pidx], ycnts[pidx]
    
    clist = np.array([np.sum(ycnts[:(x+1)]) for x in range(len(ycnts))])
    vv = np.where((clist > npix*0.05) & (clist < npix*0.95))[0]
    a, b = np.polyfit(xbins[vv], clist[vv], 1)
    x1, x2 = -b/a, (len(values)-b)/a
    # plt.figure()
    # plt.plot(xbins[vv], clist[vv], 'ko-')
    # plt.plot([x1, x2], [0, len(values)], 'r-', lw=3, alpha=0.5)
    # plt.savefig('c_dist_%i_%i' % (xpeak, sigv))
    
    return x1, x2 


def find_centroids(data_array, x_low, x_upper, y_low, y_upper, mean, std, burn_limit, star_std, std_limit):

    x_upper = int(min(x_upper, len(data_array[0])))
    y_upper = int(min(y_upper, len(data_array)))
    x_low = int(max(0, x_low))
    y_low = int(max(0, y_low))

    data_array = np.full_like(data_array[y_low:y_upper + 1, x_low:x_upper + 1],
                              data_array[y_low:y_upper + 1, x_low:x_upper + 1])

    test = []

    for i in range(-star_std, star_std + 1):
        for j in range(-star_std, star_std + 1):
            rolled = np.roll(np.roll(data_array, i, 0), j, 1)
            test.append(rolled)

    median_test = np.median(test, 0)
    max_test = np.max(test, 0)
    del test
    stars = np.where((data_array < burn_limit) & (data_array > mean + std_limit * std) & (max_test == data_array)
                     & (median_test > mean + 2 * std))
    del data_array

    stars = [stars[1] + x_low, stars[0] + y_low]
    stars = np.swapaxes(stars, 0, 1)

    return stars
