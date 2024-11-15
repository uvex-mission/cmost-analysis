''' 
    Functions relating to opening and manipulating CMOST images
'''
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
        
    graycode : bool
        Defaults to False
        If True, applies gray code descrambling to the cds frames
        
    Methods
    -------
    read_fits
    
    perform_cds
    
    bin_frames
    
    get_info
    
    cleanup_frames
    
    descramble_gray_code
    
    Attributes
    ----------
    filepath : string
        Location of FITS image file within data directory
    
    readout_mode : string 
        Camera readout mode used
    
    date : datetime 
        Date and time of when image was taken
    
    exp_time : float
        Exposure time in seconds
    
    led_voltage : float
        LED voltage in Volts
    
    temperature : float
        Detector temperature
    
    camera_id : string
        ID of the CMOST setup being used
    
    det_id : string
        ID of the detector being used
        
    dev_size : tuple
        Device size in pixels as (width, height)
        Will instead store the guiding row size for GUIDING readout mode
    
    gain : string
        Gain mode being used
    
    firmware : string
        Controller firmware being used
    
    tpixel_hold : int
        TPixel Hold
    
    custom_key_values: dict
        Dictionary of custom keys and associated values
    
    raw_frames : 3-d uint32 array
        A Numpy array of 2-d frames, containing the recorded pixel values
    
    cds_frames : 3-d or 4-d float64 array
        A Numpy array of 2-d frames, containing the pixel values after CDS processing has been applied
        If HDR more is used, array is shape len(raw_frames) x 2 x frame shape,
        and for each raw_frame, cds_frames contains a high-gain frame followed by a low-gain frame
    
    binned_frames : 3-d or 4-d float64 array
        A Numpy array of 2-d frames, containing CDS frames that have been binned up by a factor supplied
        by the user (default 4)
    '''
    def __init__(self, filepath='', custom_keys=[], cleanup=True, graycode=False):
        self.filepath = filepath
    
        if self.filepath != '':
            # Read image at provided filepath
            self.read_fits(self.filepath, custom_keys, graycode)
        
        # Once everything is loaded delete what's no longer needed
        if cleanup:
            self.cleanup_frames()

    def read_fits(self, filepath, custom_keys, graycode):
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
        self.led_voltage = float(cmost_hdr.get('LED',-1))
        self.temperature = float(cmost_hdr.get('TEMP',-1))
        # Do exposure time math so that everything is in seconds
        if cmost_hdr.get('LONGEXPO') == 1:
            # Exposure time in s
            self.exp_time = float(cmost_hdr.get('EXPTIME',-1))
        else:
            # Exposure time in ms, convert to s
            self.exp_time = float(cmost_hdr.get('EXPTIME',-1000)) / 1000.
        self.camera_id = cmost_hdr.get('CAMERAID','')
        self.det_id = cmost_hdr.get('DETID','')
        if self.readout_mode in ['ROLLINGRESET_HDR']:
            self.gain = 'hdr' # Gain header is not always consistently set for HDR mode
        else:
            self.gain = cmost_hdr.get('GAIN','')
        self.firmware = cmost_hdr.get('FIRMWARE','')
        self.tpixel_hold = cmost_hdr.get('TPIXEL_H',-1)
    
        # Any other non-default header keys to search for as passed by user
        self.custom_key_values = {}
        for k in custom_keys:
            self.custom_key_values[k] = cmost_hdr.get(k,None)
    
        if len(cmost_file) > 1:
            # Create an array of useable frames
            frame_shape = cmost_file[1].data.shape
            # This file will have at least one unusable frame
            if self.readout_mode in ['DEFAULT','ROLLINGRESET','ROLLINGRESET_HDR']:
                if cmost_hdr.get('NORESET',0) == 1:
                    # Reset frame not saved, just ignore 0th Extension
                    ignore_ext = 1
                else:
                    # Ignore 0th extension and first frame (dummy reset frame is meaningless)
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
            # Frame data is in uint16 or uint32 by default, open in uint32
            self.raw_frames[i] = np.array(cmost_file[i+ignore_ext].data, dtype=np.uint32)

        # Perform CDS on the frames
        self.perform_cds()
        
        # Perform gray code descrambling
        if graycode:
            self.descramble_gray_code()

        cmost_file.close()

    def perform_cds(self):
        '''
        Perform necessary CDS operation based on the readout mode on all useable frames
        '''
        col_width = 256 # Column width in pixels
        if self.readout_mode in ['DEFAULT','ROLLINGRESET','PSEUDOGLOBALRESET','GUIDING']:
            # CDS columns are laid out side by side width-wise
            oldshape = self.raw_frames.shape
            AmpNb = oldshape[2]//(col_width*2) # Number of amplifiers
            image = np.reshape(self.raw_frames,
                            (oldshape[0], oldshape[1]*col_width, 2, AmpNb),
                            order='F')
            cds_image = image[:,:,0,:] - image[:,:,1,:]
            self.cds_frames = np.reshape(cds_image,
                            (oldshape[0], oldshape[1], oldshape[2]//2),
                            order='F')
            self.dev_size = (self.cds_frames.shape[2], self.cds_frames.shape[1]) # Width, height
        elif self.readout_mode in ['TRUEGLOBALRESET']:
            # CDS columns are laid out top and bottom
            oldshape = self.raw_frames.shape
            AmpHt = oldshape[1]//2 # Amplifier height
            self.cds_frames = self.raw_frames[:,AmpHt:,:] - self.raw_frames[:,:AmpHt,:]
            self.dev_size = (self.cds_frames.shape[2], self.cds_frames.shape[1]) # Width, height
        elif self.readout_mode in ['ROLLINGRESET_HDR']:
            # Four columns, order is 0: Reset Low, 1: Reset High, 2: Signal High, 3: Signal Low
            oldshape = self.raw_frames.shape
            AmpNb = oldshape[2]//(col_width*4) # Number of amplifiers
            image = np.reshape(self.raw_frames,
                            (oldshape[0], oldshape[1]*col_width, 4, AmpNb),
                            order='F')
            cds_high = image[:,:,1,:] - image[:,:,2,:]
            cds_low = image[:,:,0,:] - image[:,:,3,:]

            cds_high_frames = np.reshape(cds_high, (oldshape[0], oldshape[1],
                            oldshape[2]//4), order='F')
            cds_low_frames = np.reshape(cds_low, (oldshape[0], oldshape[1],
                            oldshape[2]//4), order='F')
            # 2-d array of 2-d frames, with shape len(raw_frames) x 2
            self.cds_frames = np.stack((cds_high_frames, cds_low_frames), axis=1)
            self.dev_size = (self.cds_frames.shape[3], self.cds_frames.shape[2]) # Width, height
        else:
            # Just return the raw image
            self.cds_frames = self.raw_frames
            self.dev_size = (self.cds_frames.shape[2], self.cds_frames.shape[1]) # Width, height
    
    def descramble_gray_code(self):
        '''
        Perform gray code descrambling on CDS frames
        '''
        def binary_to_gray(n):
            n = int(n)
            n ^= (n>>1)
            return n
        
        col_width = 256
        orig_shape = self.cds_frames.shape
        
        binarycode = range(col_width)
        graycode = np.array([binary_to_gray(i) for i in binarycode])
        t = np.argsort(graycode, axis=-1, kind=None, order=None)
        
        # Sort CDS frames into correct order
        if len(orig_shape) == 3: # Single-gain mode frame
            AmpNb = orig_shape[2]//col_width # Number of amplifiers
            image = np.reshape(self.cds_frames,(orig_shape[0],orig_shape[1],col_width,AmpNb), order='F')
            image = image[:,:,t,:]
            self.cds_frames = image.reshape(orig_shape, order='F')
        elif len(orig_shape) == 4: # Dual-gain mode frame
            AmpNb = orig_shape[3]//col_width
            image = np.reshape(self.cds_frames,(orig_shape[0],orig_shape[1],orig_shape[2],col_width,AmpNb), order='F')
            image = image[:,:,:,t,:]
            self.cds_frames = image.reshape(orig_shape, order='F')

    def bin_frames(self, bin_size=4):
        '''
        Bin frames by given number of pixels (defaults to 4x4)
        '''
        if self.readout_mode in ['ROLLINGRESET_HDR']:
            print('ERROR: binning not yet implemented for HDR mode, get Hannah to do this')
        elif (self.cds_frames.shape[1] % bin_size == 0) & (self.cds_frames.shape[2] % bin_size == 0):
            self.binned_frames = self.cds_frames.reshape(self.cds_frames.shape[0],
                                    self.cds_frames.shape[1] // bin_size, bin_size,
                                    self.cds_frames.shape[2] // bin_size, bin_size).mean(-1).mean(2)
        else:
            print('ERROR: bin_size must be exact divisor of {}, {}'.format(self.cds_frames.shape[1],
                                                                            self.cds_frames.shape[2]))

        return self.binned_frames

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
        
        device_string = ''
        if self.readout_mode in ['GUIDING']:
            device_string = 'Guiding Row'
        else:
            device_string = 'Device'

        # Create info string
        info_string = """ Properties:
        Firmware: {}
        Camera ID: {}
        Detector ID: {}
        {} Size: {} x {} pixels
        Date: {}
        Readout mode: {}
        Exposure time: {} s
        LED voltage: {} V
        Temperature: {} K
        TPixel Hold: {}
        Gain mode: {}
        Number of frames: {}{} frames
        {}
        """.format(self.firmware, self.camera_id, self.det_id,
                    device_string, self.dev_size[0], self.dev_size[1],
                    self.date.isoformat(), self.readout_mode, self.exp_time,
                    self.led_voltage, self.temperature, self.tpixel_hold,
                    self.gain, len(self.cds_frames), hdr_string, custom_key_str)
        return info_string

    def get_mean(self, subframe, mask=None):
        '''
        Calculate mean of subframe across all useable frames
        
        Parameters
        ----------
        subframe : array, list or tuple
            Indices of the subframe in format (x1, x2, y1, y2)
            
        mask : 2-D int array
            Array of same dimensions as subframe to mask bad pixels
            Where mask > 0, pixel will be masked from calculation

        Returns
        -------
        m : float or float array
            Mean of subframe across all useable frames, or array of means for HDR mode
        '''
        x1, x2, y1, y2 = subframe
        if self.readout_mode in ['ROLLINGRESET_HDR']:
            if mask is not None:
                mask = np.stack([mask]*self.cds_frames.shape[1])
                mask = np.stack([mask]*self.cds_frames.shape[0])
                masked_cds_frames = np.ma.masked_where(mask > 0, self.cds_frames[:,:,y1:y2,x1:x2])
                m = np.ma.mean(masked_cds_frames,axis=(0,2,3))
            else:
                m = np.mean(self.cds_frames[:,:,y1:y2,x1:x2],axis=(0,2,3))
        else:
            if mask is not None:
                mask = np.stack([mask]*self.cds_frames.shape[0])
                masked_cds_frames = np.ma.masked_where(mask > 0, self.cds_frames[:,y1:y2,x1:x2])
                m = np.ma.mean(masked_cds_frames)
            else:
                m = np.mean(self.cds_frames[:,y1:y2,x1:x2])
        return m

    def get_median(self, subframe, mask=None):
        '''
        Calculate median of subframe across all useable frames
        
        Parameters
        ----------
        subframe : array, list or tuple
            Indices of the subframe in format (x1, x2, y1, y2)
            
        mask : 2-D int array
            Array of same dimensions as subframe to mask bad pixels
            Where mask > 0, pixel will be masked from calculation

        Returns
        -------
        m : float or float array
            Median of subframe across all useable frames, or array of medians for HDR mode
        '''
        x1, x2, y1, y2 = subframe
        if self.readout_mode in ['ROLLINGRESET_HDR']:
            if mask is not None:
                mask = np.stack([mask]*self.cds_frames.shape[1])
                mask = np.stack([mask]*self.cds_frames.shape[0])
                masked_cds_frames = np.ma.masked_where(mask > 0, self.cds_frames[:,:,y1:y2,x1:x2])
                m = np.ma.median(masked_cds_frames,axis=(0,2,3))
            else:
                m = np.median(self.cds_frames[:,:,y1:y2,x1:x2],axis=(0,2,3))
        else:
            if mask is not None:
                mask = np.stack([mask]*self.cds_frames.shape[0])
                masked_cds_frames = np.ma.masked_where(mask > 0, self.cds_frames[:,y1:y2,x1:x2])
                m = np.ma.median(masked_cds_frames)
            else:
                m = np.median(self.cds_frames[:,y1:y2,x1:x2])
        return m

    def get_variance(self, subframe, mask=None):
        '''
        Calculate variance of subframe of difference between first two useable frames

        Parameters
        ----------
        subframe : array, list or tuple
            Indices of the subframe in format (x1, x2, y1, y2)
            
        mask : 2-D int array
            Array of same dimensions as subframe to mask bad pixels
            Where mask > 0, pixel will be masked from calculation

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
                if mask is not None:
                    mask = np.stack([mask]*diff.shape[0])
                    masked_diff = np.ma.masked_where(mask > 0, diff)
                    var = np.ma.var(masked_diff,axis=(1,2)) / 2
                else:
                    var = np.var(diff,axis=(1,2)) / 2
            else:
                diff = self.cds_frames[1,y1:y2,x1:x2] - self.cds_frames[0,y1:y2,x1:x2]
                if mask is not None:
                    masked_diff = np.ma.masked_where(mask > 0, diff)
                    var = np.ma.var(masked_diff) / 2
                else:
                    var = np.var(diff) / 2
            return var
        else:
            print('WARNING: Not enough frames to calculate variance')
            if self.readout_mode in ['ROLLINGRESET_HDR']:
                var = np.array([0, 0])
            else:
                var = 0
            return var

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
    if os.path.isdir(file_prefix):
        directory = file_prefix
        
        filepaths = [os.path.join(directory, f) for f in os.listdir(directory)]
    else:
        directory = os.path.dirname(file_prefix)
        prefix = os.path.basename(file_prefix)
    
        filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if prefix in f]

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
    if len(os.listdir(directory)) > 0:
        table_rows = []
        for f in os.listdir(directory):
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath):
                # Open fits file to view header
                try:
                    cmost_file = fits.open(filepath)
                    hdr = cmost_file[0].header
                    hdu = len(cmost_file)
                    cmost_file.close()
                except:
                    print("Couldn't open {}, ignoring".format(filepath))
                    continue
                    
                # Set gain key word
                readout_mode = hdr.get('READOUTM','DEFAULT')
                if hdr.get('READOUTM') in ['ROLLINGRESET_HDR']:
                    gain = 'hdr'
                else:
                    gain = hdr.get('GAIN','')
                    
                # Do exposure time math so that everything is in seconds
                if hdr.get('LONGEXPO') == 1:
                    # Exposure time in s
                    exp_time = float(hdr.get('EXPTIME',-1))
                else:
                    # Exposure time in ms
                    exp_time = float(hdr.get('EXPTIME',-1000)) / 1000.
                
                # Get number of exposures in this file
                if readout_mode in ['DEFAULT','ROLLINGRESET','ROLLINGRESET_HDR']:
                    if hdr.get('NORESET',0) == 1:
                        # Reset frame has not been recorded
                        num_exp = hdu - 1
                    else:
                        # Ignore 0th extension and first frame (data is meaningless)
                        num_exp = hdu - 2
                else:
                    # Just ignore 0th Extension
                    num_exp = hdu - 1
                    
                # LED should be a voltage i.e. float but has sometimes been used inconsistently
                led = hdr.get('LED',-1)
                try:
                    led = float(led)
                except:
                    # Handle safely if it's a non-numeric string
                    led = -1.
                
                # List the default properties
                row = [filepath, readout_mode,
                        hdr.get('DATE','0001-01-01T00:00:00'),
                        exp_time, led,
                        float(hdr.get('TEMP',-1)), hdr.get('CAMERAID',''),
                        hdr.get('DETID',''), gain, hdr.get('FIRMWARE',''),
                        float(hdr.get('TPIXEL_H',-1)), num_exp]
                
                # Add in any custom keys
                for k in custom_keys:
                    # If key not found, default to None
                    row.append(hdr.get(k,None))
                
                # Create table row
                table_rows.append(row)
        
        if len(table_rows) == 0:
            print(f'No files in {directory}')
            return False
        
        # Define column names
        col_names = ['FILEPATH', 'READOUTM', 'DATE', 'EXPTIME', 'LED', 'TEMP',
                    'CAMERAID', 'DETID', 'GAIN', 'FIRMWARE', 'TPIXEL_H', 'NUM_EXP']
        col_names.extend(custom_keys)
        
        # Construct astropy table
        t = Table(rows=table_rows, names=col_names)
        
        return t
    
    else:
        print(f'No files in {directory}')
        return False

