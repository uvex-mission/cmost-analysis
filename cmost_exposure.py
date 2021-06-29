''' Functions relating to opening and manipulating CMOST images'''
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from datetime import datetime

class Exposure():
	'''
	Exposure object contains data from FITS file and various derived properties
	
    Parameters
    ----------
    filepath : string
        Defaults to ''
        If provided, populate attributes from the FITS image at data_dir/filepath
        
    custom_keys : string array or list
    	Defaults to []
    	Specific header keys to load in addition to the defaults
        
    cleanup : bool
    	Defaults to True
    	If True, delete raw frames after CDS to save on memory
        
    Methods
    -------
    read_fits
    
    perform_cds
    
    get_info
    
    cleanup_frames
    
    Attributes
    ----------
    filepath : string
    	Location of FITS image file within data directory
    	
    readout_mode : string 
    	Camera readout mode used
    	
    date : datetime 
    	Date and time of when image was taken
        
    exp_time : float
    	Exposure time in milliseconds
    	
    led_voltage : float
    	LED voltage in Volts
    	
    temperature : float
    	Detector temperature
    	
    camera_id : string
    	ID of the CMOST setup being used
    	
    det_id : string
    	ID of the detector being used
    	
    gain : string
    	Gain mode being used
    	
    custom_key_values: dict
    	Dictionary of custom keys and associated values
    	
    raw_frames : 3-d int32 array
    	A Numpy array of 2-d frames, containing the recorded pixel values
    	
    cds_frames : 3-d or 4-d int32 array
    	A Numpy array of 2-d frames, containing the pixel values after CDS processing has been applied
    	If HDR more is used, array is shape len(raw_frames) x 2 x frame shape, 
    	and for each raw_frame, cds_frames contains a high-gain frame followed by a low-gain frame
	'''
	def __init__(self, filepath='', custom_keys=[], cleanup=True):
		self.filepath = filepath
		
		if self.filepath != '':
			# Read image at provided filepath
			self.read_fits(self.filepath, custom_keys)
		
		# Once everything is loaded delete what's no longer needed
		if cleanup:
			self.cleanup_frames()

	def read_fits(self, filepath, custom_keys):
		'''
		Read FITS image and populate attributes
		
		
        Parameters
        ----------
        filepath : string
    		Location of FITS image file within data directory
		'''
		self.filepath = filepath
		
		cmost_file = fits.open(filepath)
		cmost_hdr = cmost_file[0].header

		# DATE and READOUTM are set automatically by the camera control
		self.readout_mode = cmost_hdr.get('READOUTM','DEFAULT')
		self.date = datetime.fromisoformat(cmost_hdr.get('DATE','0001-01-01'))

		# Other headers are set by the user, so cope if they are not set
		self.exp_time = float(cmost_hdr.get('EXPTIME',-1))
		self.led_voltage = float(cmost_hdr.get('LED',-1))
		self.temperature = float(cmost_hdr.get('TEMP',-1))
		self.camera_id = cmost_hdr.get('CAMERAID','')
		self.det_id = cmost_hdr.get('DETID','')
		self.gain = cmost_hdr.get('GAIN','')
		
		# Any other non-default header keys to search for as passed by user
		self.custom_key_values = {}
		for k in custom_keys:
			self.custom_key_values[k] = cmost_hdr.get(k,None)
		
		if len(cmost_file) > 1:
			# Create an array of useable frames
			frame_shape = cmost_file[1].data.shape	
		
			# This file will have at least one unusable frame
			if self.readout_mode in ['DEFAULT','ROLLINGRESET','ROLLINGRESET_HDR']:
				# Ignore 0th extension and first frame (data is meaningless)
				ignore_ext = 2
			else:
				# Just ignore 0th Extension
				ignore_ext = 1
		else:
			# Image is stored in the 0th Extension
			# For frames not taken directly from the camera
			ignore_ext = 0
			frame_shape = cmost_file[0].data.shape
		
		useable_frames = len(cmost_file) - ignore_ext
		
		self.raw_frames = np.zeros([useable_frames,frame_shape[0],frame_shape[1]])
		for i in range(useable_frames):
			# Frame data is in uint16 by default, open in int32
			self.raw_frames[i] = np.array(cmost_file[i+ignore_ext].data, dtype=np.int32)
			
		# Perform CDS on the frames
		self.perform_cds()

		cmost_file.close()

	def perform_cds(self):
		'''
		Perform necessary CDS operation based on the readout mode on all useable frames
		'''
		if self.readout_mode in ['DEFAULT','ROLLINGRESET','PSEUDOGLOBALRESET']:
			# CDS columns are laid out side by side
			oldshape = self.raw_frames.shape
			image = np.reshape(self.raw_frames, 
							(oldshape[0], oldshape[1]*256, 2, oldshape[2]//256//2),
							order='F')
			cds_image = image[:,:,0,:] - image[:,:,1,:]
			self.cds_frames = np.reshape(cds_image, 
							(oldshape[0], oldshape[1], oldshape[2]//2), 
							order='F')
		elif self.readout_mode in ['TRUEGLOBALRESET']:
			# CDS columns are laid out top and bottom
			self.cds_frames = self.raw_frames[:,2048:,:] - self.raw_frames[:,:2048,:]
		elif self.readout_mode in ['ROLLINGRESET_HDR']:
			# Four columns, order is 0: Reset Low, 1: Reset High, 2: Signal High, 3: Signal Low
			oldshape = self.raw_frames.shape
			image = np.reshape(self.raw_frames,(oldshape[0], oldshape[1]*256, 4, 
							oldshape[2]//256//4),order='F')
			
			cds_high = image[:,:,1,:] - image[:,:,2,:]
			cds_low = image[:,:,0,:] - image[:,:,3,:]

			cds_high_frames = np.reshape(cds_high, (oldshape[0], oldshape[1], 
							oldshape[2]//4), order='F')
			cds_low_frames = np.reshape(cds_low, (oldshape[0], oldshape[1], 
							oldshape[2]//4), order='F')
			# 2-d array of 2-d frames, with shape len(raw_frames) x 2
			self.cds_frames = np.stack((cds_high_frames, cds_low_frames), axis=1)
		else:
			# Just return the raw image
			self.cds_frames = self.raw_frames

	def get_info(self):
		'''
		Output useful image info in a nice readable format
		
		Returns
		-------
		info_string : string
			String containing properties from FITS header and number of frames
		'''
		# Build custom keys string
		custom_key_str = ''
		for k in self.custom_key_values:
			custom_key_str += '{}: {}\n\t\t'.format(k,self.custom_key_values[k])
			
		hdr_string = ''
		if self.readout_mode in ['ROLLINGRESET_HDR']:
			hdr_string = ' x 2'
		
		# Create info string
		info_string = """ Properties: 
		Readout mode: {} 
		Date: {} 
		Exposure time: {} ms
		LED voltage: {} V 
		Temperature: {} K
		Camera ID: {} 
		Detector ID: {}
		Gain mode: {}
		Number of frames: {}{} frames
		{}
		""".format(self.readout_mode, self.date.isoformat(), self.exp_time,
					self.led_voltage, self.temperature, self.camera_id, self.det_id,
					self.gain, len(self.cds_frames), hdr_string, custom_key_str)
		return info_string

	def get_mean(self, subframe):
		'''
		Calculate mean of subframe across all useable frames
		
		Parameters
		----------
		subframe : array, list or tuple
			Indices of the subframe in format (x1, x2, y1, y2)
			
		Returns
		-------
		m : float or float array
			Mean of subframe across all useable frames, or array of means for HDR mode
		'''
		x1, x2, y1, y2 = subframe
		if self.readout_mode in ['ROLLINGRESET_HDR']:
			m = np.mean(self.cds_frames[:,:,y1:y2,x1:x2],axis=(0,2,3))
		else:
			m = np.mean(self.cds_frames[:,y1:y2,x1:x2])
		return m
	
	def get_median(self, subframe):
		'''
		Calculate median of subframe across all useable frames
		
		Parameters
		----------
		subframe : array, list or tuple
			Indices of the subframe in format (x1, x2, y1, y2)
			
		Returns
		-------
		m : float or float array
			Median of subframe across all useable frames, or array of medians for HDR mode
		'''
		x1, x2, y1, y2 = subframe
		if self.readout_mode in ['ROLLINGRESET_HDR']:
			m = np.median(self.cds_frames[:,:,y1:y2,x1:x2],axis=(0,2,3))
		else:
			m = np.median(self.cds_frames[:,y1:y2,x1:x2])
		return m
	
	def get_variance(self, subframe):
		'''
		Calculate variance of subframe of difference between first two useable frames
		
		Parameters
		----------
		subframe : array, list or tuple
			Indices of the subframe in format (x1, x2, y1, y2)
			
		Returns
		-------
		var : float or float array
			Variance of the subframe if more than one useable frame, otherwise 0.
			Array of variances for HDR mode
		'''
		x1, x2, y1, y2 = subframe
		if len(self.cds_frames) > 1:
			if self.readout_mode in ['ROLLINGRESET_HDR']:
				diff = self.cds_frames[1,:,y1:y2,x1:x2] - self.cds_frames[0,:,y1:y2,x1:x2]
				var = np.var(diff,axis=(1,2)) / 2
			else:
				diff = self.cds_frames[1,y1:y2,x1:x2] - self.cds_frames[0,y1:y2,x1:x2]
				var = np.var(diff) / 2
			return var
		else:
			print('WARNING: Not enough frames to calculate variance')
			return 0
		
	def cleanup_frames(self):
		'''
		Delete raw frames to save on memory
		'''
		self.raw_frames = np.array([])
		

'''
	Utility functions for loading Exposure objects
'''

def load_by_file_prefix(file_prefix,**kwargs):
	'''
	Load all FITS files that begin with the filepath string provided
	
	Parameters
    ----------
    file_prefix : string
    	Common path to desired FITS files, including filename prefix
    	
    **kwargs
    	Any keyword arguments to pass on to Exposure
    	
    Returns
    -------
    List of Exposure objects
	'''
	directory = os.path.dirname(file_prefix)
	filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
	
	return load_by_filepath(filepaths,**kwargs)
	
def load_by_filepath(filepaths,**kwargs):
	'''
	Load FITS files from filepaths provided
	
	Parameters
    ----------
    filepaths : str array
    	Array of filepaths to desired FITS files
    	
    **kwargs
    	Any keyword arguments to pass on to Exposure
    	
    Returns
    -------
    List of Exposure objects
	'''
	exposures = [Exposure(f,**kwargs) for f in filepaths]
	return exposures

	
def scan_headers(directory,custom_keys=[]):
	'''
	Scan the FITS headers in given directory and return file details
	
	Parameters
    ----------
    directory : string
    	Path to the directory to scan
    	
    custom_keys : string array or list
    	Defaults to []
    	Specific non-default header keys to search
    	
    Returns
    -------
    Astropy Table of FITS header contents of files in given directory 
    '''	
	table_rows = []
	for f in os.listdir(directory):
		filepath = os.path.join(directory, f)
		if os.path.isfile(filepath):
    		# Open fits file to view header
			cmost_file = fits.open(filepath)
			hdr = cmost_file[0].header
			cmost_file.close()
			
			# List the default properties
			row = [filepath, hdr['READOUTM'], datetime.fromisoformat(hdr['DATE']),
					float(hdr.get('EXPTIME',-1)), float(hdr.get('LED',-1)), 
					float(hdr.get('TEMP',-1)), hdr.get('CAMERAID',''), 
					hdr.get('DETID',''), hdr.get('GAIN','')]
			
			# Add in any custom keys
			for k in custom_keys:
				# If key not found, default to None
				row.append(hdr.get(k,None))
    		
			# Create table row
			table_rows.append(row)
	
	# Define column names
	col_names = ['FILEPATH', 'READOUTM', 'DATE', 'EXPTIME', 'LED', 'TEMP', 
					'CAMERAID', 'DETID', 'GAIN']
	col_names.extend(custom_keys)
	
	# Construct astropy table
	t = Table(rows=table_rows, names=col_names)
	
	return t

