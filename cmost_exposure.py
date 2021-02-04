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
        Defaults to ''.
        If provided, populate attributes from the FITS image at data_dir/filepath
        
    Methods
    -------
    read_fits
    
    perform_cds
    
    get_info
    
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
    	
    raw_frames : 3-d int32 array
    	A Numpy array of 2-d frames, containing the recorded pixel values
    	
    cds_frames : 3-d int32 array
    	A Numpy array of 2-d frames, containing the pixel values after CDS processing has been applied
    
	'''
	def __init__(self, filepath=''):
		self.filepath = filepath
		
		if self.filepath != '':
			# Read image at provided filepath
			self.read_fits(self.filepath)
		
		# Subframe


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
		self.exp_time = float(cmost_hdr.get('EXPTIME',-1))
		self.led_voltage = float(cmost_hdr.get('LED',-1))
		self.temperature = float(cmost_hdr.get('TEMP',-1))
		self.camera_id = cmost_hdr.get('CAMERAID','')
		self.det_id = cmost_hdr.get('DETID','')
		self.gain = cmost_hdr.get('GAIN','')

		# Create an array of useable frames
		frame_shape = cmost_file[1].data.shape
		# Ignore 0th extension and first frame (data is meaningless)
		useable_frames = len(cmost_file)-2 
		
		self.raw_frames = np.zeros([useable_frames,frame_shape[0],frame_shape[1]])
		for i in range(useable_frames):
			# Frame data is in uint16 by default, open in int32
			self.raw_frames[i] = np.array(cmost_file[i+2].data, dtype=np.int32)
			
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
			oldshape = self.raw_frames.shape
			image = np.reshape(self.raw_frames,
							(oldshape[0], oldshape[1]//2, 2, oldshape[2]),
							order='F')
			cds_image = image[:,:,1,:] - image[:,:,0,:]
			self.cds_frames = cds_image		
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
		info_string = """ Properties: 
		Readout mode: {} 
		Date: {} 
		Exposure time: {} ms
		LED voltage: {} V 
		Temperature: {} K
		Camera ID: {} 
		Detector ID: {}
		Gain mode: {}
		Number of frames: {}
		""".format(self.readout_mode, self.date.isoformat(), self.exp_time,
					self.led_voltage, self.temperature, self.camera_id, self.det_id,
					self.gain, len(self.raw_frames))
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
		Mean of subframe across all useable frames
		'''
		x1, x2, y1, y2 = subframe
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
		Median of subframe across all useable frames
		'''
		x1, x2, y1, y2 = subframe
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
		Variance of the subframe if more than one useable frame, otherwise 0
		'''
		x1, x2, y1, y2 = subframe
		if len(self.cds_frames) > 1:
			diff = self.cds_frames[1,y1:y2,x1:x2] - self.cds_frames[0,y1:y2,x1:x2]
			var = np.var(diff) / 2
			return var
		else:
			print('WARNING: Not enough frames to calculate variance')
			return 0

'''
	Utility functions for loading Exposure objects
'''

def load_by_file_prefix(file_prefix):
	'''
	Load all FITS files that begin with the filepath string provided
	
	Parameters
    ----------
    file_prefix : string
    	Common path to desired FITS files, including filename prefix
    	
    Returns
    -------
    List of Exposure objects
	'''
	directory = os.path.dirname(file_prefix)
	filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
	
	return load_by_filepath(filepaths)
	
def load_by_filepath(filepaths):
	'''
	Load FITS files from filepaths provided
	
	Parameters
    ----------
    filepaths : str array
    	Array of filepaths to desired FITS files
    	
    Returns
    -------
    List of Exposure objects
	'''
	exposures = [Exposure(f) for f in filepaths[:-1]]
	return exposures

	
def scan_headers(directory):
	'''
	Scan the FITS headers in given directory and return file details
	
	Parameters
    ----------
    directory : string
    	Path to the directory to scan
    	
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
    		
			# Create table row
			table_rows.append((filepath, hdr['READOUTM'], datetime.fromisoformat(hdr['DATE']),
							float(hdr.get('EXPTIME',-1)), float(hdr.get('LED',-1)), 
							float(hdr.get('TEMP',-1)), hdr.get('CAMERAID',''), 
							hdr.get('DETID',''), hdr.get('GAIN','')))
	
	# Construct astropy table
	t = Table(rows=table_rows, 
				names=('FILEPATH', 'READOUTM', 'DATE', 'EXPTIME', 'LED', 'TEMP', 'CAMERAID', 'DETID', 'GAIN'))
	
	return t
