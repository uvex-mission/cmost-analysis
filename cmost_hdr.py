''' Functions relating to HDR mode '''
import numpy as np
import astropy.units as u
import astropy.constants as cr
from astropy.modeling.blackbody import FLAM
from astropy.convolution import convolve_fft, Gaussian2DKernel

# Set telescope details
epd = 75 * u.cm
area = np.pi * (0.5*epd)**2
reflectivity = 0.9
mirrors = 3
qe = 0.4 * u.electron / u.ph

# Band-specific details
band_wav = {'nuv': [200, 300] * u.nm, 'fuv': [145, 175] * u.nm}
sky_bgd_rate = {'nuv': 1.0 * u.ph / u.s, 'fuv': 0.015 * u.ph / u.s} # per pixel
dichroic = {'nuv': 0.75, 'fuv': 0.5}

# Set some detector details
psf_pix = 4
dark_current = 0.01 * u.electron / u.s
pixel_size = 1 * u.arcsec

# Gain-mode-specific details
gain = {'high': 1.2 * u.adu / u.electron, 'low': 10.1 * u.adu / u.electron}
read_noise = {'high': 2 * u.electron, 'low': 10 * u.electron}
well_depth = {'high': 25000 * u.electron, 'low': 190000 * u.electron}

# PSF details
psf_fwhm = 2 * u.arcsec

def magnitude_to_count_rate(magnitudes, band='fuv'):
	''' Convert (flat) input magnitudes to count rates
	
	Parameters
	----------
	magnitudes : float array
		Array of magnitudes in ABmag units
		
	band : string
		Which UVEX band to use, options are 'nuv' and 'fuv'
		Defaults to 'fuv'
		
	Returns
	-------
	count_rate : float array
		Array of count rates in ph/s units
	'''
	wav = np.arange(1000,5000) * u.AA # Wavelength scale in 1 Angstrom steps
	dw = 1 * u.AA
	ph_energy = (cr.h.cgs * cr.c.cgs / wav.cgs) / u.ph

	count_rate = np.zeros(len(magnitudes)) * u.ph / u.s
	for i, m in enumerate(magnitudes):
	
		# Convert to flux density
		flux_den = m.to(FLAM, equivalencies=u.spectral_density(wav))
		ph_flux = flux_den * dw / ph_energy

		# In-band rate
		fluence = ph_flux[(wav >= band_wav[band][0].to(u.AA)) & (wav <= band_wav[band][1].to(u.AA))].sum()

		count_rate[i] = fluence * area * (reflectivity**mirrors) * dichroic[band]

	return count_rate
	
def count_rate_to_electron_rate(count_rate):
	''' Calculates the number of measured electrons for given count rate and exposure time
	
	Parameters
	----------
	count_rate : float array
		Array of count rates in photons per second
		
	exp_time: float
		Exposure time in seconds
		
	gain_mode : string
		Which detector gain mode to use, options are 'low' and 'high'
		Defaults to 'high'
		
	Returns
	-------
	e : float array
		Array of number of count rate  in electrons per second the same shape as count_rate
	'''
	e = count_rate * qe
	# No quantum yield consideration at this time
	
	return e
	
def count_rate_to_electrons(count_rate, exp_time, gain_mode='high'):
	''' Calculates the number of measured electrons for given count rate and exposure time
	Incorporates dark current and saturation, but otherwise returns ideal number of electrons (no noise/quantization)
	
	Parameters
	----------
	count_rate : float array
		Array of count rates in photons per second
		
	exp_time: float
		Exposure time in seconds
		
	gain_mode : string
		Which detector gain mode to use, options are 'low' and 'high'
		Defaults to 'high'
		
	Returns
	-------
	e : float array
		Array of number of electrons measured the same shape as count_rate
		
	saturated : bool array
		Array the same shape as count_rate stating which pixels are saturated (i.e. True) 
	'''
	e = (count_rate * qe) * exp_time 
	# No quantum yield consideration at this time

	# Handle saturation
	saturated = (e + dark_current * exp_time) > well_depth[gain_mode]
	e[saturated] = well_depth[gain_mode]
	
	return e, saturated
	
	
def get_signal(count_rate, exp_times, gain_mode='high'):
	''' Runs count_rate_to_electrons then converts to detector signal
	
	Parameters
	----------
	count_rate : float array
		Array of count rates in photons per second
		
	exp_times: float array
		Exposure time in seconds
		
	gain_mode : string
		Which detector gain mode to use, options are 'low' and 'high'
		Defaults to 'high'
		
	Returns
	-------
	signal : 2-D float array
		Array of detector signal in ADU with shape len(exp_times) x len(count_rate)
		
	saturated : 2-D bool array
		Array stating which pixels are saturated (i.e. True) with shape len(exp_times) x len(count_rate)
	'''
	signals, sat = [], []
	for t in exp_times:
		e, saturated = count_rate_to_electrons(count_rate, t, gain_mode=gain_mode)
		signals.append(e * gain[gain_mode])
		sat.append(saturated)
	
	return np.array(signals), np.array(sat)
	
	
def get_snr(count_rate, exp_times, band='fuv', gain_mode='high'):
	''' Calculate SNR for given count rates and exposure times
	
	Parameters
	----------
	count_rate : float array
		Array of count rates in photons per second
		
	exp_times: float array
		Array of exposure times in seconds
		
	band : string
		Which UVEX band to use, options are 'nuv' and 'fuv'
		Defaults to 'fuv'
		
	gain_mode : string
		Which detector gain mode to use, options are 'low' and 'high'
		Defaults to 'high'
		
	Returns
	-------
	snr : 2-D float array
		Array of SNR with shape len(exp_times) x len(count_rate)
	'''
	snr_list = []
	for t in exp_times:
		# Input photon flux to electrons for given exposure time
		signal, saturated = count_rate_to_electrons(count_rate, t, gain_mode=gain_mode)

		# Get background rate
		sky_bgd, _ = count_rate_to_electrons(sky_bgd_rate[band], t, gain_mode=gain_mode)

		# Calculate shot noise and dark noise
		shot_noise = np.sqrt(signal + sky_bgd).value * u.electron 

		# Get SNR
		snr = signal / np.sqrt(shot_noise**2 + read_noise[gain_mode]**2)
		snr[saturated] = 0
		
		snr_list.append(snr) 
		# Also add Fano noise in quad with these when implementing quantum yield
	
	return np.array(snr_list)


def perform_hdr_simple(pixels, saturated, exp_times):
	''' Simple HDR algorithm to iterate through exposure times and select pixels from as long an exposure as possible
	
	Parameters
	----------
	pixels : 2-D float array
		Array of pixel values (could be signal or SNR) to be selected
		Second dimension must be same length as exp_times
		
	saturated: bool array
		Array stating which pixels are saturated (i.e. True), the same shape as pixels
		
	exp_times: float array
		Array of exposure times in seconds
		
	Returns
	-------
	hdr_pixels : float array
		1-D array of pixel values selected from pixels based on exp_times and saturation
	'''
	# Sort exp_times by duration, longest to shortest
	sortind = np.argsort(-exp_times)
	
	hdr_pixels = np.zeros(pixels.shape[1])
	prev_sat = True
	for i in sortind:
		this_sat = saturated[i]
		
		# If pixel was saturated in a previous exposure but not this one, assign value
		this_exp = prev_sat & ~this_sat
		hdr_pixels[this_exp] = pixels[i][this_exp]
		
		prev_sat = this_sat
	
	return hdr_pixels


def create_image(im_frame_size,exp_time,sources=[],band='fuv',gain_mode='high'):
    ''' Creates an image from an exposure, with given sources
        
    Parameters
    ----------
    im_frame_size : int
        Size of the resulting image in pixels
        
    exp_time: float
        Exposure time in seconds
        
    sources: QTable object
        QTable of sources in format (x_pos, y_pos, count_rate)
        x_pos and y_pos are floats, fractional positions between 0 and 1
        count_rate is in photons per second
    '''
    
    # Initialise oversampling
    oversample = 6
    src_frame_size = 25 # In pixels
    pixel_size_init = pixel_size / oversample
    src_frame_size_init = src_frame_size * oversample
    im_frame_size_init = im_frame_size * oversample

    # Create empty oversampled image
    im_array = np.zeros([im_frame_size_init,im_frame_size_init]) * u.ph / u.s

    # Create PSF kernel
    psf_kernel = Gaussian2DKernel(psf_fwhm / pixel_size_init,
                                  x_size=src_frame_size_init, y_size=src_frame_size_init)
    psf_array = psf_kernel.array

    # Add sources
    if len(sources) > 0:
        source_inv = np.array([sources['y_pos'],sources['x_pos']]) # Create array of all ys and all xs
        source_pix = (source_inv.transpose() * np.array(im_array.shape)).transpose().astype(int)
        im_array[tuple(source_pix)] += sources['src_cr']

    # Now convolve with the PSF
    im_psf = convolve_fft(im_array.value, psf_kernel) * im_array.unit

    # Bin up the image by oversample parameter to the correct pixel size
    shape = (im_frame_size, oversample, im_frame_size, oversample)
    im_binned = im_psf.reshape(shape).sum(-1).sum(1)
    im_binned[im_binned < 0] = 0

    # Convert to observed source counts
    im_counts = im_binned * exp_time
    im_sky = np.ones(im_counts.shape) * sky_bgd_rate[band] * exp_time

    # Observe! Includes sky rate and dark current
    im_poisson = (np.random.poisson(im_counts.value) + np.random.poisson(im_sky.value)) * im_counts.unit

    # Read! Convert to electrons, apply saturation and read noise
    # No quantum yield consideration at this time
    im_read = im_poisson * qe + dark_current * exp_time
    im_read[im_read > well_depth[gain_mode]] = well_depth[gain_mode]
    im_read += np.random.normal(loc=0, scale=read_noise[gain_mode].value,
                                size=im_read.shape) * im_read.unit

    im_read = np.floor(im_read)
    im_read[im_read < 0] = 0
    
    # Finally, convert to ADU
    im_adu = im_read * gain[gain_mode]

    return im_adu

