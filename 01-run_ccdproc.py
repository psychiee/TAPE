# -*- coding: utf-8 -*-
"""
01-run_ccdproc 
 (1) make the list of images (BIAS, DARK, FLAT, OBJECT)
 (2) generate the MASTER frames (BIAS, DARK, FLAT)
 (3) do preprocessing OBJECT by (BIAS, DARK, FLAT) (from NYSC 1-m telescope)
 
@author: wskang
@update: 2019/09/25
"""
import os, time
from glob import glob
import numpy as np 
from astropy.io import fits 
from photlib import read_params, prnlog

# READ the parameter file
par = read_params() 

# MOVE to the working directory
os.chdir(par['WORKDIR'])
prnlog('#WORK DIR: %s' % par['WORKDIR'])
# FILTERING for binning option
BINNING = int(par['BINNING'])
prnlog('#BINNING for processing: %i' % BINNING)
# MAKE object + bias + dark + flat image file list 
# modify wild cards (*.fits) for your observation file name
flist = glob('object-*.fits')  
flist = flist + glob('dark-*.fits') 
flist = flist + glob('flat-*.fits') 
flist = flist + glob('bias-*.fits') 

# DEFINE the list for information of FITS files 
TARGET, TYPE, DATEOBS, EXPTIME, FILTER, FNAME = [],[],[],[],[],[]
for fname in flist: 
    hdu = fits.open(fname)
    img, hdr = hdu[0].data, hdu[0].header
    if hdr.get('XBINNING') != BINNING: continue
    if hdr.get('YBINNING') != BINNING: continue
    # READ the FITS file header and INPUT into the list 
    TARGET.append(hdr['OBJECT'])
    TYPE.append(str.lower(hdr['IMAGETYP']))
    DATEOBS.append(hdr['DATE-OBS'])
    EXPTIME.append(hdr['EXPTIME'])
    FILTER.append(hdr['FILTER'])
    FNAME.append(fname)

# SORT the files for the observation time 
sort_list = np.argsort(DATEOBS) 
# LOOP for the all FITS images and 
# GENERATE the file list for each group 
for rname in glob('w*.list'):
    os.remove(rname)

for s in sort_list:
    prnlog('{} {} {} {} {}'.format(DATEOBS[s], TYPE[s], FILTER[s], EXPTIME[s], TARGET[s]))
    # DEFINE the name of list file with FITS header info.  
    if TYPE[s] == 'bias': 
        lname = 'wbias.list'
    elif TYPE[s] == 'dark': 
        lname = 'wdark%is.list' % (EXPTIME[s],)
    elif TYPE[s] == 'flat':
        lname = 'wflat%s.list' % (FILTER[s],) 
    elif TYPE[s] == 'object':
        lname = 'wobj.list'
    # WRITE the FITS file name into the LIST file 
    f = open(lname, 'a') 
    f.write(FNAME[s]+'\n')
    f.close()
    prnlog('add to '+lname+' ...' )
    
time.sleep(2)    

# Make master bias ========================================================

bias_list = np.genfromtxt('wbias.list',dtype='U') 
bias_stack = []
for fname in bias_list:
    hdu = fits.open(fname)[0]
    dat, hdr = hdu.data, hdu.header
    bias_stack.append(dat)
    prnlog('%s %8.1f %8.1f %8.1f %8.1f ' % \
         (fname, np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
master_bias = np.median(bias_stack, axis=0) 
dat = master_bias
prnlog('python %8.1f %8.1f %8.1f %8.1f' % \
      (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))

prnlog('Save to wbias.fits ...')
hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
hdr.set('OBJECT', 'wbias')
fits.writeto('wbias.fits', master_bias, hdr, overwrite=True)
    
# make master dark ========================================================
list_files = glob('wdark*.list')
master_darks, exptime_darks = [], [] 
for lname in list_files:
    dark_list = np.genfromtxt(lname, dtype='U') 
    fidx = os.path.splitext(lname)[0]

    dark_stack = [] 
    for fname in dark_list: 
        hdu = fits.open(fname)[0]
        dat, hdr = hdu.data, hdu.header
        dark_stack.append(dat - master_bias) 
        prnlog('%s %8.1f %8.1f %8.1f %8.1f ' % \
           (fname, np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    master_dark = np.median(dark_stack, axis=0)
    exptime_dark = hdr.get('EXPTIME')
    prnlog('Save to %s.fits ...%s %i' % (fidx, hdr['IMAGETYP'], hdr['EXPTIME']))
    hdr.set('OBJECT',fidx)
    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    hdr.set('EXPTIME', exptime_dark)
    fits.writeto(fidx+'.fits', master_dark, hdr, overwrite=True)        

    dat = master_dark 
    prnlog('Python %8.1f %8.1f %8.1f %8.1f' % \
           (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))

    master_darks.append(master_dark)
    exptime_darks.append(exptime_dark)


# make master flats =======================================================
list_files = glob('wflat*.list') 
master_flats, filter_flats = [], [] 
for lname in list_files: 
    flat_list = np.genfromtxt(lname, dtype='U') 
    fidx = os.path.splitext(lname)[0]

    flat_stack = [] 
    for fname in flat_list:
        hdu = fits.open(fname)[0]
        dat, hdr = hdu.data, hdu.header
        dat = dat - master_bias
        fdat = dat / np.median(dat)
        flat_stack.append(fdat)
        filter_flat = str.strip(hdr.get('FILTER'))
        prnlog('%s %8.1f %8.1f %8.1f %8.1f ' % \
            (fname, np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    master_flat = np.median(flat_stack, axis=0)
    prnlog('Save to %s.fits ...%s %s' % (fidx, hdr['IMAGETYP'], hdr['FILTER']))
    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    hdr.set('OBJECT', fidx)
    hdr.set('FILTER', filter_flat)
    fits.writeto(fidx+'.fits', master_flat, hdr, overwrite=True)         
        
    dat = master_flat
    prnlog('Python %8.6f %8.6f %8.6f %8.6f' % \
           (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    master_flats.append(master_flat)
    filter_flats.append(filter_flat)    

# Do preprocessing of object images 
flist = np.genfromtxt('wobj.list', dtype='U')
for fname in flist:
    hdu = fits.open(fname)[0]
    cIMAGE, hdr = hdu.data, hdu.header
    cFILTER = str.strip(hdr.get('FILTER'))
    cEXPTIME = float(hdr.get('EXPTIME'))

    # Find closest exposure time dark
    dd = np.argmin(np.abs(np.array(exptime_darks) - cEXPTIME))
    dEXPTIME = exptime_darks[dd]
    dFRAME = master_darks[dd]
    dFRAME = dFRAME * (cEXPTIME/dEXPTIME)
    # Find flat image
    ff = filter_flats.index(cFILTER)
    fFILTER = filter_flats[ff]
    fFRAME = master_flats[ff]
    cIMAGE = cIMAGE - master_bias - dFRAME
    cIMAGE = cIMAGE / fFRAME

    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    fits.writeto('w'+fname, cIMAGE, hdr, overwrite=True)

    prnlog('{}({},{})<<[{},{}]'.format(fname, cFILTER, cEXPTIME, fFILTER, dEXPTIME))

