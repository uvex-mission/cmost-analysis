''' Functions relating to opening and manipulating CMOST images'''
from __future__ import division, print_function
import os
import re
import numpy as np
from astropy.io import fits
from datetime import datetime

class Exposure():
	'''
	Exposure object contains data from FITS file and various derived properties
	
    Parameters
    ----------
    filepath : string
        Defaults to ''.
        If provided, populate attributes from the FITS image at data_dir/filepath
        
    Methods
    -------
    read_fits
    
    perform_cds
    
    print_info
    
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
    	
    raw_frames : 3-d int32 array
    	A Numpy array of 2-d frames, containing the recorded pixel values
    	
    cds_frames : 3-d int32 array
    	A Numpy array of 2-d frames, containing the pixel values after CDS processing has been applied
    
	'''
	def __init__(self, filepath=''):
		self.filepath = filepath
		
		# Initialize empty attributes
		self.readout_mode = ''
		self.date = datetime.fromtimestamp(0)
		self.exp_time = -1.
		self.led_voltage = -1.
		self.temperature = -1.
		self.raw_frames = np.array([])
		
		if self.filepath != '':
			# Read image at provided filepath
			self.read_fits(self.filepath)

		# Perform CDS on frames
		if len(self.raw_frames) > 0:
			self.perform_cds()
		
		# CamID and DetID? Gain mode?


	def read_fits(self,filepath):
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
		self.readout_mode = cmost_hdr['READOUTM']
		self.date = datetime.fromisoformat(cmost_hdr['DATE'])

		# Other headers are set by the user, so cope if they are not set
		if 'EXPTIME' in cmost_hdr.keys(): 
			self.exp_time = float(cmost_hdr['EXPTIME'])
		if 'LED' in cmost_hdr.keys(): 
			self.led_voltage = float(cmost_hdr['LED'])
		if 'TEMP' in cmost_hdr.keys(): 
			self.temperature = float(cmost_hdr['TEMP'])

		# Create an array of useable frames
		frame_shape = cmost_file[1].data.shape
		# Ignore 0th extension and first frame (data is meaningless)
		useable_frames = len(cmost_file)-2 
		
		self.raw_frames = np.zeros([useable_frames,frame_shape[0],frame_shape[1]])
		for i in range(useable_frames):
			# Frame data is in uint16 by default, open in int32
			self.raw_frames[i] = np.array(cmost_file[i+2].data, dtype=np.int32)

		cmost_file.close()

	def perform_cds(self):
		'''
		Perform necessary CDS operation based on the readout mode
		'''
		x = 'To-do'

	def print_info():
		'''
		Output useful image info in a nice readable format
		'''
		x = 'To-do' 




# To-do: 

#def get_frames_by_fileprefix(fileprefix):
# Behaviour as in Hannah's code
	
#def get_frames_by_filenames(filenames):
# Pass an array of filenames, return list of objects
	
#def scan_headers():
# Scan headers of FITS files in the directory you specify
# Return a an astropy Table of header contents 

