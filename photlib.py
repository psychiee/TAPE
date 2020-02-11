#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:19:01 2018

@author: wskang
"""
import numpy as np 
import scipy.ndimage as sn
import scipy.optimize as so
from astropy import time, coordinates as coord
from astropy.stats import sigma_clipped_stats
#from photutils import DAOStarFinder 

LAT, LON, H = 34.5261362, 127.4470482, 81.35789

def read_params():
    f = open('pyapw.par', 'r')
    par = {}
    for line in f: 
        tmp = line.split()
        par.update({tmp[0]:tmp[1]})
    return par 

def sigma_clip(xx, lower=3, upper=3):
    '''
    mxx = np.ma.masked_array(xx)
    while 1:
        avg = np.mean(mxx)
        sig = np.std(mxx)
        mask = (mxx <= (avg - lower*sig)) | (mxx > (avg + upper*sig))
        if (mask == mxx.mask).all():
            break
        mxx.mask = mask
    return np.median(mxx), np.std(mxx)
    '''
    return sigma_clipped_stats(xx, sigma_lower=lower, sigma_upper=upper)

def helio_JD(DATEOBS, RA, Dec, exptime=0):
    objcoo = coord.SkyCoord(RA, Dec, unit=('hourangle', 'deg'), frame='fk5')
    doao = coord.EarthLocation.from_geodetic(LON, LAT, H)
    times = time.Time(DATEOBS, format='isot', scale='utc', location=doao)
    dt = time.TimeDelta(exptime / 2.0, format='sec')
    times = times + dt
    # ltt_bary = times.light_travel_time(objcoo)
    # times_bary = times.tdb + ltt_bary
    ltt_helio = times.light_travel_time(objcoo, 'heliocentric')
    times_helio = times.utc + ltt_helio

    return times_helio.jd

#==================================================================
# FUNCTIONS for aperture photometry 
#==================================================================
def detect_peaks(image, detection_area = 2):
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
    neighborhood = np.ones((1+2*detection_area,1+2*detection_area))
    #apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
    local_max = sn.maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are looking for, but also the background. In order to isolate the peaks we must remove the background from the mask. We create the mask of the background
    background = (image==0)
    #a little technicality: we must erode the background in order to successfully subtract it from local_max, otherwise a line will appear along the background border (artifact of the local maximum filter)
    eroded_background = sn.binary_erosion(background, structure=neighborhood, border_value=1)
    #we obtain the final mask, containing only peaks, by removing the background from the local_max mask
    #detected_peaks = local_max - eroded_background
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

    #--- threshold ---
    imageT = image>=threshold

    #--- smooth threshold, mask and filter initial image ---
    neighborhood = sn.generate_binary_structure(2,1)
    imageT = sn.binary_erosion(imageT, structure=neighborhood, border_value=1)
    imageT = sn.binary_dilation(imageT, structure=neighborhood, border_value=1)
    imageF = sn.filters.gaussian_filter(imageT*image,sigmafilter)

    #--- find local max ---
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

    #--- removes too bright stars (possibly saturated) ---
    if saturation[0]:
        peaks = peaks * (image<saturation[1])
    
    return peaks

    
def find_stars(image, nb_stars, excluded = [], included = [], detection_area = 2, saturation = [False,0], margin = 5, text_display = True):
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

    #--- included ---
    if len(included)>0:
        x0, x1, y0, y1 = included
        im = image[x0:x1,y0:y1]
    else:
        im = image
        x0, y0 = [0,0]
    
    #--- get subimage information ---
    maxi = np.max(im.flatten())
    #median = stats.get_median(im)[0]
    medi = np.median(im)
        
    #--- searching threshold ---
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
    
    #---------------------------------------------------------------------------------
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    #if star is saturated it could be that median value is 32767 or 65535 --> height=0
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

    #---------------------------------------------------------------------------------
    #fitting gaussian
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
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    beta = p[7]
    
    #angle gives the direction of the p[4]=fwhm_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):
        fwhm_large = np.abs(p[4])
        fwhm_small = np.abs(p[5])
        angle = np.arctan(np.tan(p[6]))
    else:    #then fwhm_1 is the smallest : we want angle to point to sig_y, the largest
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

#############################################################################
# ZSCALING
#############################################################################

MAX_REJECT = 0.5
MIN_NPIXELS = 5
GOOD_PIXEL = 0
BAD_PIXEL = 1
KREJ = 2.5
MAX_ITERATIONS = 5

def zscale(image, nsamples=1000, contrast=0.25, bpmask=None, zmask=None):
    """Implement IRAF zscale algorithm
    nsamples=1000 and contrast=0.25 are the IRAF display task defaults
    bpmask and zmask not implemented yet
    image is a 2-d numpy array
    returns (z1, z2)
    """

    # Sample the image
    samples = zsc_sample (image, nsamples, bpmask, zmask)
    npix = len(samples)
    samples.sort()
    zmin = samples[0]
    zmax = samples[-1]
    # For a zero-indexed array
    center_pixel = (npix - 1) // 2
    if npix%2 == 1:
        median = samples[center_pixel]
    else:
        median = 0.5 * (samples[center_pixel] + samples[center_pixel + 1])

    #
    # Fit a line to the sorted array of samples
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    ngrow = max (1, int (npix * 0.01))
    ngoodpix, zstart, zslope = zsc_fit_line (samples, npix, KREJ, ngrow,
                                             MAX_ITERATIONS)

    if ngoodpix < minpix:
        z1 = zmin
        z2 = zmax
    else:
        if contrast > 0: zslope = zslope / contrast
        z1 = max (zmin, median - (center_pixel - 1) * zslope)
        z2 = min (zmax, median + (npix - center_pixel) * zslope)
    return z1, z2

def zsc_sample (image, maxpix, bpmask=None, zmask=None):
   
    # Figure out which pixels to use for the zscale algorithm
    # Returns the 1-d array samples
    # Don't worry about the bad pixel mask or zmask for the moment
    # Sample in a square grid, and return the first maxpix in the sample
    nc = image.shape[0]
    nl = image.shape[1]
    stride = max (1.0, np.sqrt((nc - 1) * (nl - 1) / float(maxpix)))
    stride = int (stride)
    samples = image[::stride,::stride].flatten()
    return samples[:maxpix]
   
def zsc_fit_line (samples, npix, krej, ngrow, maxiter):

    #
    # First re-map indices from -1.0 to 1.0
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix)
    xnorm = xnorm * xscale - 1.0

    ngoodpix = npix
    minpix = max (MIN_NPIXELS, int (npix*MAX_REJECT))
    last_ngoodpix = npix + 1

    # This is the mask used in k-sigma clipping.  0 is good, 1 is bad
    badpix = np.zeros(npix, dtype="int32")

    #
    #  Iterate

    for niter in range(maxiter):

        if (ngoodpix >= last_ngoodpix) or (ngoodpix < minpix):
            break
       
        # Accumulate sums to calculate straight line fit
        goodpixels = np.where(badpix == GOOD_PIXEL)
        sumx = xnorm[goodpixels].sum()
        sumxx = (xnorm[goodpixels]*xnorm[goodpixels]).sum()
        sumxy = (xnorm[goodpixels]*samples[goodpixels]).sum()
        sumy = samples[goodpixels].sum()
        sum = len(goodpixels[0])

        delta = sum * sumxx - sumx * sumx
        # Slope and intercept
        intercept = (sumxx * sumy - sumx * sumxy) / delta
        slope = (sum * sumxy - sumx * sumy) / delta
       
        # Subtract fitted line from the data array
        fitted = xnorm*slope + intercept
        flat = samples - fitted
    
        # Compute the k-sigma rejection threshold
        ngoodpix, mean, sigma = zsc_compute_sigma (flat, badpix, npix)
    
        threshold = sigma * krej
    
        # Detect and reject pixels further than k*sigma from the fitted line
        lcut = -threshold
        hcut = threshold
        below = np.where(flat < lcut)
        above = np.where(flat > hcut)
    
        badpix[below] = BAD_PIXEL
        badpix[above] = BAD_PIXEL
           
        # Convolve with a kernel of length ngrow
        kernel = np.ones(ngrow,dtype="int32")
        badpix = np.convolve(badpix, kernel, mode='same')
    
        ngoodpix = len(np.where(badpix == GOOD_PIXEL)[0])
           
        niter += 1
    
    # Transform the line coefficients back to the X range [0:npix-1]
    zstart = intercept - slope
    zslope = slope * xscale
    
    return ngoodpix, zstart, zslope

def zsc_compute_sigma (flat, badpix, npix):
    
    # Compute the rms deviation from the mean of a flattened array.
    # Ignore rejected pixels
    # Accumulate sum and sum of squares
    goodpixels = np.where(badpix == GOOD_PIXEL)
    sumz = flat[goodpixels].sum()
    sumsq = (flat[goodpixels]*flat[goodpixels]).sum()
    ngoodpix = len(goodpixels[0])
    if ngoodpix == 0:
        mean = None
        sigma = None
    elif ngoodpix == 1:
        mean = sumz
        sigma = None
    else:
        mean = sumz / ngoodpix
        temp = sumsq / (ngoodpix - 1) - sumz*sumz / (ngoodpix * (ngoodpix - 1))
        if temp < 0:
            sigma = 0.0
        else:
            sigma = np.sqrt (temp)
    
    return ngoodpix, mean, sigma

