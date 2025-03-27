'''
    Functions relating to analysing the standard analysis exposures
    taken using the cmost_camera.py scripts. Operates on the same
    file naming scheme as cmost_camera.py so if that changes, this
    needs to change too
    
    Run this with Python 3
    
    Usage: python cmost_analysis.py [-g] [-s 0,1023,0,4095] data/20240413
'''
import os, sys
sys.path.append('..')
import numpy as np
import time
from datetime import datetime
import subprocess
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import fitting
from astropy.modeling.models import PowerLaw1D
from astropy.convolution import convolve, Box2DKernel
from astropy.table import Table
from cmost_exposure import Exposure, load_by_file_prefix, load_by_filepath, scan_headers
from scipy.ndimage import gaussian_filter
from scipy import signal

font = {'size' : 12, 'family' : 'sans-serif'}
matplotlib.rc('font', **font)

def standard_analysis_products(dirname, **kwargs):
    '''
    Load standard exposures from the given directory and output
    standard analysis reports
    '''
    
    #######################################################
    # Prep section
    # - Parses available files and gets general information
    #######################################################
    
    # Scan the FITS files in the data directory
    file_table = scan_headers(dirname,custom_keys=['LEDWAVE'])
    if not file_table:
        exit('No data files found in directory, closing')
    
    # Get camera and detector ID
    f = os.path.split(file_table[0]['FILEPATH'])[1].split('_')
    camera = f[0]
    detid = f[1]
    
    # Load first file that's not a guiding, noisespectrum or bias file (otherwise doesn't matter which it is) for device information
    non_guiding = [(('guiding' not in f) & ('bias' not in f) & ('noisespec' not in f)) for f in file_table['FILEPATH']]
    if np.sum(non_guiding) == 0:
        # TODO: If we only have bias frames, make a way to get exp info without actually loading the frame
        non_guiding = [(('guiding' not in f) & ('noisespec' not in f)) for f in file_table['FILEPATH']]
    exp = Exposure(file_table[non_guiding][0]['FILEPATH'])
    cw = exp.col_width
    n_channels = exp.dev_size[0] // cw
    
    # Scan through files to find out what's available
    bias_present, noise_present, opdark_present, longdark_present, singleframe_present = 0, 0, 0, 0, 0
    flat_present, flatdark_present, persist_present, notes_present = 0, 0, 0, 0
    opdark_modes = []
    for f in file_table['FILEPATH']:
        if 'bias' in f: bias_present = 1
        if 'noisespec' in f: noise_present = 1
        if 'NUVdark' in f:
            if 'NUVdark' not in opdark_modes: opdark_modes.append('NUVdark')
            opdark_present = 1
        if 'NUVguidingdark' in f:
            if 'NUVguidingdark' not in opdark_modes: opdark_modes.append('NUVguidingdark')
            opdark_present = 1
        if 'FUVdark' in f:
            if 'FUVdark' not in opdark_modes: opdark_modes.append('FUVdark')
            opdark_present = 1
        if 'longdark' in f: longdark_present = 1
        if 'flat_' in f: flat_present = 1
        if 'flatdark' in f: flatdark_present = 1
        if 'singleframe' in f: singleframe_present = 1
        if 'persist' in f: persist_present = 1

    # Check for notes file
    if os.path.exists(dirname+'/analysis_notes.txt'):
        notes_present = 1
        notes_file = open(dirname+'/analysis_notes.txt')
        notes_lines = notes_file.readlines()
        notes_file.close()
        
        # Get datestring from first line of analysis notes
        datestring = notes_lines[0].split()[3]
        filedatestring = datestring[0:4]+datestring[5:7]+datestring[8:10]
    else:
        # Guess datestring from the file name
        d = exp.filepath.split('_')
        datestring = d[-1][0:4]+'-'+d[-1][4:6]+'-'+d[-1][6:8]
        filedatestring = d[-1][0:8]
        
    # Check for LED used from exposures taken under illumination
    is_ledwave = file_table['LEDWAVE'] != None
    if np.sum(is_ledwave) > 0:
        ledwave = np.unique(file_table['LEDWAVE'][is_ledwave].value)
    else:
        ledwave = None

    # Final summary numbers to generate
    read_noise, det_gain, use_gain, dark_current, well_depth, read_time = {}, {}, {}, {}, {}, {}
    read_noise_e, dark_current_e, well_depth_e = {}, {}, {}
    noise_bad, dark_bad, flat_bad = {}, {}, {}
    noise_percent_bad, dark_percent_bad, flat_percent_bad = {}, {}, {}
    gain_modes = ['high (dual-gain)','low (dual-gain)','high','low']
    # A nominal gain value for converting ADU to electrons in case PTC gives bad values
    nom_gain = {'high (dual-gain)': 1.2, 'low (dual-gain)': 8.5, 'high': 1.2, 'low': 8.5}
    nom_read_noise = {'high (dual-gain)': 1.75, 'low (dual-gain)': 1.18, 'high': 1.75, 'low': 1.18}
    
    # Create a folder to hold output of analysis run inside the provided data directory
    nowstring = time.strftime("%Y%m%d%H%M%S")
    output_dirname = os.path.join(dirname,f'{nowstring}_output')
    os.mkdir(output_dirname)
    output_prefix = f'{filedatestring}_{camera}_{detid}'
    
    # Initialize bad pixel map and create summary filename based on whether or not we're using a subframe
    if 'subframe' in kwargs:
        subframe = kwargs['subframe']
        x, y = subframe[1]-subframe[0]+1, subframe[3]-subframe[2]+1
        bad_pixel_map = np.zeros([x,y])
        doc_name = f'{output_prefix}{subframe}_analysis_report_{nowstring}.pdf'
    else:
        bad_pixel_map = np.zeros([exp.dev_size[1],exp.dev_size[0]])
        doc_name = f'{output_prefix}_analysis_report_{nowstring}.pdf'
        
    # Set the definitions of bad pixels
    # Define bad pixels as having noise above 3e- threshold in high gain
    noise_bad_thresh = 3
    # Define bad pixels as having dark current exceeding 3e-3 e/s
    dark_bad_thresh = 0.003
    # Define bad pixels as being 5-sigma below flat mean (set later)
    flat_bad_thresh = {}
    
    ###################################################
    # Data analysis section
    # - Performs analysis tasks and writes result files
    ###################################################
    
    # If no frame time, get frame readout time from the single-frame exposures
    if (exp.frame_time < 0) and singleframe_present and notes_present:
        for i, l in enumerate(notes_lines):
            if l.startswith('Readout times:'):
                # A single frame readout lasts for a reset frame readout and an actual frame readout
                # So the actual time an exposure takes reading out is more-or-less half this amount
                read_time['high'] = float(notes_lines[i+1].split()[1]) / 2
                read_time['low'] = float(notes_lines[i+2].split()[1]) / 2
                read_time['hdr'] = float(notes_lines[i+3].split()[1]) / 2
    
    # Get noise spectrum
    if noise_present:
        nsfr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_noisespec', custom_keys=['XPIX','YPIX'], **kwargs)
        noise_freq, noise_spectrum, noise_pix = [], [], []
        for ns in nsfr:
            noise_frame = ns.cds_frames[0]
            noise_pix.append([ns.custom_key_values['XPIX'],ns.custom_key_values['YPIX']])
            
            # reshape the raw image into time series for each amplifier
            noise_gain = 65.8e-6 # in uV/DN, currently hardcoded
            data_q = np.reshape(noise_frame,(noise_frame.shape[0],int(noise_frame.shape[1]/n_channels),n_channels),order='F') * noise_gain
            pixel_timeseries = np.reshape(data_q,(data_q.shape[0]*data_q.shape[1],data_q.shape[2]),order='c') # Time series

            # Compute the FFT using Welch's method for a smooth plot
            sample_rate = 11.1e6 # in Hz
            freq, pxx_den = signal.welch(pixel_timeseries,fs=sample_rate,axis=0,nperseg=2**16)
            noise_freq.append(freq)
            noise_spectrum.append(np.sqrt(pxx_den))
    
    # Get bias frames, noise maps, and read noise measurements
    if bias_present:
        # Loop through the gain modes and load each bias frame individually using subframes
        # These files are *very* large
        med_bias_frames, noise_map, bias_comment, noise_comment = {}, {}, {}, {}
        med_bias_frames_e, noise_map_e = {}, {}
        for gain in ['hdr','high','low']:
            if 'subframe' in kwargs:
                # If a subframe is already defined, perform this on the subframe
                bifr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias_{gain}', **kwargs)[0]
                n_frames = len(bifr.cds_frames)

                # Build the median bias and noise frames
                if gain == 'hdr':
                    med_bias_frames['high (dual-gain)'] = np.nanmedian(bifr.cds_frames[:,0],axis=0)
                    med_bias_frames['low (dual-gain)'] = np.nanmedian(bifr.cds_frames[:,1],axis=0)
                    noise_map['high (dual-gain)'] = np.nanstd(bifr.cds_frames[:,0],axis=0)
                    noise_map['low (dual-gain)'] = np.nanstd(bifr.cds_frames[:,1],axis=0)
                    bias_comment['high (dual-gain)'] = f'Median of {n_frames} minimum-length high gain (dual-gain) exposures'
                    bias_comment['low (dual-gain)'] = f'Median of {n_frames} minimum-length low gain (dual-gain) exposures'
                    noise_comment['high (dual-gain)'] = f'Noise map over {n_frames} minimum-length high gain (dual-gain) exposures'
                    noise_comment['low (dual-gain)'] = f'Noise map over {n_frames} minimum-length low gain (dual-gain) exposures'
                else:
                    med_bias_frames[gain] = np.nanmedian(bifr.cds_frames,axis=0)
                    noise_map[gain] = np.nanstd(bifr.cds_frames,axis=0)
                    bias_comment[gain] = f'Median of {n_frames} minimum-length {gain} gain exposures'
                    noise_comment[gain] = f'Noise map over {n_frames} minimum-length {gain} gain exposures'
            else:
                # If doing this for the whole detector, use subframe functionality
                # to do this channel-by-channel to save time and memory
                if gain == 'hdr':
                    med_bias_frames['high (dual-gain)'] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    med_bias_frames['low (dual-gain)'] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    noise_map['high (dual-gain)'] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    noise_map['low (dual-gain)'] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    for i in range(n_channels):
                        chsf = (i*cw,(i+1)*cw-1,0,exp.dev_size[1]-1) # Define subframe for this channel
                        bifr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias_{gain}', subframe=chsf, **kwargs)[0]
                        
                        # Build the median bias and noise frames
                        med_bias_frames['high (dual-gain)'][:,i*cw:(i+1)*cw] = np.nanmedian(bifr.cds_frames[:,0],axis=0)
                        med_bias_frames['low (dual-gain)'][:,i*cw:(i+1)*cw] = np.nanmedian(bifr.cds_frames[:,1],axis=0)
                        noise_map['high (dual-gain)'][:,i*cw:(i+1)*cw] = np.nanstd(bifr.cds_frames[:,0],axis=0)
                        noise_map['low (dual-gain)'][:,i*cw:(i+1)*cw] = np.nanstd(bifr.cds_frames[:,1],axis=0)
                    n_frames = len(bifr.cds_frames)
                    
                    # Store number of frames used
                    bias_comment['high (dual-gain)'] = f'Median of {n_frames} minimum-length high gain (dual-gain) exposures'
                    bias_comment['low (dual-gain)'] = f'Median of {n_frames} minimum-length low gain (dual-gain) exposures'
                    noise_comment['high (dual-gain)'] = f'Noise map over {n_frames} minimum-length high gain (dual-gain) exposures'
                    noise_comment['low (dual-gain)'] = f'Noise map over {n_frames} minimum-length low gain (dual-gain) exposures'
                else:
                    med_bias_frames[gain] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    noise_map[gain] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    for i in range(n_channels):
                        chsf = (i*cw,(i+1)*cw-1,0,exp.dev_size[1]-1) # Define subframe for this channel
                        bifr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias_{gain}', subframe=chsf, **kwargs)[0]
                        
                        # Build the median bias and noise frames
                        med_bias_frames[gain][:,i*cw:(i+1)*cw] = np.nanmedian(bifr.cds_frames,axis=0)
                        noise_map[gain][:,i*cw:(i+1)*cw] = np.nanstd(bifr.cds_frames,axis=0)
                    n_frames = len(bifr.cds_frames)
                    
                    # Store number of frames used
                    bias_comment[gain] = f'Median of {n_frames} minimum-length {gain} gain exposures'
                    noise_comment[gain] = f'Noise map over {n_frames} minimum-length {gain} gain exposures'
        
        # Calculate the read noise
        for g in noise_map:
            # Calculate RMS read noise after filtering out catastrophically bad pixels
            read_noise[g] = np.sqrt(np.nanmean(noise_map[g][(noise_map[g] * nom_gain[g]) < 100]**2))
            
        # Write the median bias frames to file
        bias_outpath = write_fits_image(med_bias_frames, 'bias frame', bias_comment,
                                        os.path.join(output_dirname,f'{output_prefix}_bias_frames.fits'))
        noise_outpath = write_fits_image(noise_map, 'noise map', noise_comment,
                                         os.path.join(output_dirname,f'{output_prefix}_noise_maps.fits'))
        
        # If bias is present, bias frame will be subtracted from flats
        bias_note = ', bias frame subtracted'
    else:
        bias_note = ', no bias frame subtracted'
        
        
    if longdark_present:
        # Load long dark frames
        longdark_files = load_by_file_prefix(f'{dirname}/{camera}_{detid}_longdark', **kwargs)
        
        longdark, shortdark = {}, {}
        longdark_frames, longdark_frames_e, longdark_comment = {}, {}, {}
        long_exp_time, short_exp_time, longdark_exp_time = 0, 0, 0
        # Sort into short and long exposures by gain
        for longdark_frame in longdark_files:
            gain = longdark_frame.gain
            if gain == 'hdr':
                if longdark_frame.exp_time > 1000:
                    if 'high (dual-gain)' not in longdark: longdark['high (dual-gain)'] = []
                    if 'low (dual-gain)' not in longdark: longdark['low (dual-gain)'] = []
                
                    # Collect the actual long dark frames
                    longdark['high (dual-gain)'].append(longdark_frame.cds_frames[0,0])
                    longdark['low (dual-gain)'].append(longdark_frame.cds_frames[0,1])
                    long_exp_time = longdark_frame.exp_time
                else:
                    if 'high (dual-gain)' not in shortdark: shortdark['high (dual-gain)'] = []
                    if 'low (dual-gain)' not in shortdark: shortdark['low (dual-gain)'] = []
                
                    # Collect any shorter frames for bias subtraction
                    shortdark['high (dual-gain)'].append(longdark_frame.cds_frames[0,0])
                    shortdark['low (dual-gain)'].append(longdark_frame.cds_frames[0,1])
                    short_exp_time = longdark_frame.exp_time
            else:
                if longdark_frame.exp_time > 1000:
                    if gain not in longdark: longdark[gain] = []
                
                    # Collect the actual long dark frames
                    longdark[gain].append(longdark_frame.cds_frames[0])
                    long_exp_time = longdark_frame.exp_time
                else:
                    if gain not in shortdark: shortdark[gain] = []
                    
                    # Collect any shorter frames for bias subtraction
                    shortdark[gain].append(longdark_frame.cds_frames[0])
                    short_exp_time = longdark_frame.exp_time
        
        # Create resulting long dark images
        for g in longdark:
            long_med = np.median(np.array(longdark[g]),axis=0)
            
            # Subtract either a) a shorter dark frame or b) the bias frame or c) nothing (non-ideal scenario)
            longdark_comment[g] = f'Median of {len(longdark[g])} x {long_exp_time}s dark frames'
            if len(shortdark[g]) > 0:
                longdark_exp_time = long_exp_time - short_exp_time
                longdark_frames[g] = long_med - np.median(np.array(shortdark[g]),axis=0)
                longdark_comment[g] += f', {short_exp_time}s dark frame subtracted'
            elif bias_present:
                longdark_exp_time = long_exp_time
                longdark_frames[g] = long_med - med_bias_frames[g]
                longdark_comment[g] += bias_note
            else:
                longdark_exp_time = long_exp_time
                longdark_frames[g] = long_med
                longdark_comment[g] += bias_note
                
            # Dark current 'measurement' is the 99% percentile
            dark_current[g] = np.percentile(longdark_frames[g],99)
        
        # Write the long dark frames to file
        longdark_outpath = write_fits_image(longdark_frames, 'long dark frame', longdark_comment,
                                            os.path.join(output_dirname,f'{output_prefix}_longdark_frames.fits'))
    
    if opdark_present:
        # Loop through available operating dark modes and create data arrays
        long_high_opdark, long_low_opdark, short_high_opdark, short_low_opdark = {}, {}, {}, {}
        l_starts_opdark, s_starts_opdark = {}, {}
        oplong = {'FUVdark': 900, 'NUVdark': 300, 'NUVguidingdark': 300}
        opshort = {'FUVdark': 9, 'NUVdark': 3, 'NUVguidingdark': 3}
        for mode in opdark_modes:
            # Load the nine frames in temporal order
            short_low, short_high, long_low, long_high = [], [], [], []
            long_guideframes, short_guideframes = [], []
            s_starts, l_starts = [], []
            if mode == 'NUVguidingdark':
                # Taken using the full-dwell function, different filename format
                n_dwell, n_exp = 3, 3
                for i in range(n_dwell):
                    for j in range(n_exp):
                        # Load long and short full frames
                        long_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_FullDwell_exp{j}_UVEXNUV_4', **kwargs)[0]
                        short_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_FullDwell_exp{j}_UVEXNUV_2', **kwargs)[0]
                        
                        # No bias frame subtraction
                        long_high.append(long_frame.cds_frames[0,0])
                        long_low.append(long_frame.cds_frames[0,1])
                        short_high.append(short_frame.cds_frames[0,0])
                        short_low.append(short_frame.cds_frames[0,1])
                        
                        l_starts.append(long_frame.date)
                        s_starts.append(short_frame.date)
                        
                        # TODO: also load guiding frames
            else:
                n_frame = 9
                for i in range(n_frame):
                    long_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{oplong[mode]}_', **kwargs)[0]
                    short_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{opshort[mode]}_', **kwargs)[0]
                    
                    long_high.append(long_frame.cds_frames[0,0])
                    long_low.append(long_frame.cds_frames[0,1])
                    short_high.append(short_frame.cds_frames[0,0])
                    short_low.append(short_frame.cds_frames[0,1])
                    
                    l_starts.append(long_frame.date)
                    s_starts.append(short_frame.date)
            
            long_high_opdark[mode] = np.stack(long_high)
            long_low_opdark[mode] = np.stack(long_low)
            short_high_opdark[mode] = np.stack(short_high)
            short_low_opdark[mode] = np.stack(short_low)
            
            # Convert observation times to seconds and record
            l_starts_opdark[mode] = np.array([(l_starts[-1] - l).total_seconds() for l in l_starts])
            s_starts_opdark[mode] = np.array([(s_starts[-1] - s).total_seconds() for s in s_starts])
            
        # No frame output for this one for now
        # TODO: do we want to store median frames for these?
    
    if persist_present:
        # Load the saturated frames to confirm we're definitely saturating
        persist_sat_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_persistillum', **kwargs)[0]
        sat_voltage = persist_sat_frames.filepath.split('_')[3]
        persist_sat_median = np.nanmedian(persist_sat_frames.cds_frames, axis=0)
        
        # Load dark frames from the persistence test
        pd = load_by_file_prefix(f'{dirname}/{camera}_{detid}_persistdark', **kwargs)[0]
        
        if bias_present:
            persist_sat_median = np.nanmedian(persist_sat_frames.cds_frames, axis=0) - med_bias_frames['high']
            persist_dark_frames = pd.cds_frames - med_bias_frames['high']
        else:
            persist_sat_median = np.nanmedian(persist_sat_frames.cds_frames, axis=0)
            persist_dark_frames = pd.cds_frames
            
        if pd.frame_time > 0:
            persist_times = np.arange(len(persist_dark_frames_e)) * pd.frame_time/1000.
            persist_tu = 's'
        elif 'high' in read_time:
            persist_times = np.arange(len(persist_dark_frames_e)) * read_time['high']
            persist_tu = 's'
        else:
            persist_times = np.arange(len(persist_dark_frames_e))
            persist_tu = 'Frame number'


    if flat_present:
        # Initialize dictionaries and identify the list of flat frame files
        mid_flats, mid_flats_e, mid_flat_times, mid_flat_voltages, mid_flat_comment = {}, {}, {}, {}, {}
        flat_darks, flat_dark_times = {}, {}
        exp_times, med_sig, var, voltage = {}, {}, {}, {}
        allflats = []
        for f in file_table['FILEPATH']:
            if 'flat_' in f: allflats.append(True)
            else: allflats.append(False)
        
        # Load up example flat frames and get the means/variances for PTC/linearity plots etc.
        for gain in ['hdr','high','low']:
            # If flat darks are present, load these up
            if flatdark_present:
                flatdark_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_flatdark_{gain}', **kwargs)
                
                if gain == 'hdr':
                    flat_darks['high (dual-gain)'], flat_darks['low (dual-gain)'] = [], []
                    flat_dark_times['high (dual-gain)'], flat_dark_times['low (dual-gain)'] = [], []
                    for fd in flatdark_frames:
                        flat_darks['high (dual-gain)'].append(np.nanmedian(fd.cds_frames[:,0], axis=0))
                        flat_darks['low (dual-gain)'].append(np.nanmedian(fd.cds_frames[:,1], axis=0))
                        flat_dark_times['high (dual-gain)'].append(fd.exp_time)
                        flat_dark_times['low (dual-gain)'].append(fd.exp_time)
                else:
                    flat_darks[gain], flat_dark_times[gain] = [], []
                    for fd in flatdark_frames:
                        flat_darks[gain].append(np.nanmedian(fd.cds_frames, axis=0))
                        flat_dark_times[gain].append(fd.exp_time)
            
            # Load up the flats
            gain_flats = (file_table['GAIN'] == gain) & np.array(allflats)

            if np.sum(gain_flats) > 0:
                # Load flat frame files
                flat_files = file_table['FILEPATH'][gain_flats]
                flat_frames = load_by_filepath(file_table['FILEPATH'][gain_flats], **kwargs)
                exptime = file_table['EXPTIME'][gain_flats]
                if max(file_table['FRAMTIME'][gain_flats] > 0): exptime = exptime + file_table['FRAMTIME'][gain_flats]/1000.
                elif gain in read_time: exptime = exptime + read_time[gain]
                if (file_table['LED'][gain_flats] > -1).any():
                    led = file_table['LED'][gain_flats] # get LED voltage from FITS header
                else:
                    led = [f.split('_')[4] for f in flat_files] # get LED voltage from filename
                
                led_vals = np.unique(led)
                max_exp, max_i = max(exptime), np.argmax(exptime)
                
                # Check whether we want to do masking of existing bad pixels
                # If >50% of pixels are bad, there's probably a deeper issue, so turn bad pixel masking off
                bad_frac = np.sum((bad_pixel_map > 0)) / bad_pixel_map.size
                if bad_frac < 0.5:
                    domask = True
                    mask_str = 'masked'
                else:
                    domask = False
                    mask_str = 'not masked'
                
                # Define how to split up the detector area into multiple chunks to get a number of medians
                if 'subframe' in kwargs:
                    # If subframe defined, loop through ~roughly 256x256 chunks of the subframe
                    x_parts = int(np.maximum(1, np.round((subframe[1]-subframe[0]+1) / cw)))
                    y_parts = int(np.maximum(1, np.round((subframe[3]-subframe[2]+1) / cw)))
                    x_size, y_size = (subframe[1]-subframe[0]+1) // x_parts, (subframe[3]-subframe[2]+1) // y_parts
                else:
                    # If no subframe defined, loop through 256x256 chunks of the detector
                    x_parts, y_parts = int(exp.dev_size[0] // cw), int(exp.dev_size[1] // cw)
                    x_size, y_size = cw, cw
                n_subframes = x_parts * y_parts

                # Loop through frame pairs getting stats and mid-range frames
                if gain == 'hdr':
                    medians, variance = np.zeros((1,2)), np.zeros((1,2))
                    all_exp_times, all_voltage = np.zeros(1), np.zeros(1)
                    
                    for i in range(x_parts):
                        for j in range(y_parts):
                            x1, x2, y1, y2 = x_size*i, x_size*(i+1), y_size*j, y_size*(j+1)
                            if domask: mask = bad_pixel_map[y1:y2,x1:x2]
                            else: mask = None
                            medians = np.append(medians, [ff.get_median((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                            variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                            all_exp_times = np.append(all_exp_times, exptime)
                            all_voltage = np.append(all_voltage, led)
                    medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                    exp_times['high (dual-gain)'], exp_times['low (dual-gain)'] = all_exp_times, all_exp_times
                    voltage['high (dual-gain)'], voltage['low (dual-gain)'] = all_voltage, all_voltage
                    med_sig['high (dual-gain)'], med_sig['low (dual-gain)'] = medians[:,0], medians[:,1]
                    var['high (dual-gain)'], var['low (dual-gain)'] = variance[:,0], variance[:,1]
                    
                    # Find the mid-range exposures - something close to 10000 ADU
                    whole_frame_means = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                    hmid_exp_i = np.argmin(np.abs(whole_frame_means[:,0] - 10000))
                    lmid_exp_i = np.argmin(np.abs(whole_frame_means[:,1] - 10000))

                    mid_flat_times['high (dual-gain)'], mid_flat_times['low (dual-gain)'] = exptime[hmid_exp_i], exptime[lmid_exp_i]
                    mid_flat_voltages['high (dual-gain)'], mid_flat_voltages['low (dual-gain)'] = led[hmid_exp_i], led[lmid_exp_i]
                    if flatdark_present:
                        this_flat_darkh = flat_dark_times['high (dual-gain)'] == exptime[hmid_exp_i]
                        mid_flats['high (dual-gain)'] = flat_frames[hmid_exp_i].cds_frames[0,0] - flat_darks['high (dual-gain)'][this_flat_darkh]
                        this_flat_darkl = flat_dark_times['low (dual-gain)'] == exptime[lmid_exp_i]
                        mid_flats['low (dual-gain)'] = flat_frames[lmid_exp_i].cds_frames[0,1] - flat_darks['low (dual-gain)'][this_flat_darkl]
                    elif bias_present:
                        mid_flats['high (dual-gain)'] = flat_frames[hmid_exp_i].cds_frames[0,0] - med_bias_frames['high (dual-gain)']
                        mid_flats['low (dual-gain)'] = flat_frames[lmid_exp_i].cds_frames[0,1] - med_bias_frames['low (dual-gain)']
                    else:
                        mid_flats['high (dual-gain)'] = flat_frames[hmid_exp_i].cds_frames[0,0]
                        mid_flats['low (dual-gain)'] = flat_frames[lmid_exp_i].cds_frames[0,1]
                    mid_flat_comment['high (dual-gain)'] = f'Illuminated {exptime[hmid_exp_i]:.1f}s exposure in high (dual-gain) gain; LED at {led[hmid_exp_i]} V{bias_note}'
                    mid_flat_comment['low (dual-gain)'] = f'Illuminated {exptime[lmid_exp_i]:.1f}s exposure in low (dual-gain) gain; LED at {led[lmid_exp_i]} V{bias_note}'

                else:
                    medians, variance = np.zeros(1), np.zeros(1)
                    all_exp_times, all_voltage = np.zeros(1), np.zeros(1)
                    
                    for i in range(x_parts):
                        for j in range(y_parts):
                            x1, x2, y1, y2 = x_size*i, x_size*(i+1), y_size*j, y_size*(j+1)
                            if domask: mask = bad_pixel_map[y1:y2,x1:x2]
                            else: mask = None
                            medians = np.append(medians, [ff.get_median((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                            variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                            all_exp_times = np.append(all_exp_times, exptime)
                            all_voltage = np.append(all_voltage, led)
                    medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                    exp_times[gain] = all_exp_times
                    voltage[gain] = all_voltage
                    med_sig[gain] = medians
                    var[gain] = variance
                    
                    # Find the mid-range exposures
                    whole_frame_means = np.array([ff.get_mean((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                    mid_exp_i = np.argmin(np.abs(whole_frame_means - 10000))
                    mid_flat_times[gain] = exptime[mid_exp_i]
                    mid_flat_voltages[gain] = led[mid_exp_i]
                    if flatdark_present:
                        this_flat_dark = flat_dark_times[gain] == exptime[mid_exp_i]
                        mid_flats[gain] = flat_frames[mid_exp_i].cds_frames[0] - flat_darks[gain][this_flat_dark]
                    elif bias_present:
                        mid_flats[gain] = flat_frames[mid_exp_i].cds_frames[0] - med_bias_frames[gain]
                    else:
                        mid_flats[gain] = flat_frames[mid_exp_i].cds_frames[0]
                    mid_flat_comment[gain] = f'Illuminated {exptime[mid_exp_i]:.1f}s exposure in {gain} gain mode, LED at {led[mid_exp_i]} V{bias_note}'
        
        # Write the mid flat frames to file
        flat_outpath = write_fits_image(mid_flats, 'sample flat frame', mid_flat_comment,
                                        os.path.join(output_dirname,f'{output_prefix}_flat_frames.fits'))
        
        # Now calculate the gain
        
        # Set up fitter and model (a PL set to linear solution)
        fit = fitting.LMLSQFitter(calc_uncertainties=True)
        fit_or = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=3.5)
        model = PowerLaw1D(amplitude=1, x_0=1, alpha=-1, fixed={'x_0': True, 'alpha': True})
        
        # TODO: Move linearity fitting up here too - but first figure out what we want to do there
        
        # Fit PTC for read noise level and gain
        for_fitting, model_fit, mask_ptc = {}, {}, {}
        for g in med_sig:
            # Determine the shot noise dominated region of the PTC
            # Defined as unsaturated and above the noise-floor-dominated region
            rough_sat_level = np.minimum(max(med_sig[g]) - 5000, 10000) # We expect saturation to be well above 10k but want to handle other cases
            valid = (var[g] > 0) & (med_sig[g] > 100) & (med_sig[g] < rough_sat_level)
            
            if np.sum(valid) > 0:
                # Subtract the noise
                if g in read_noise: noise_floor = read_noise[g]**2
                else: noise_floor = nom_read_noise[g]**2
                corrected_v = var[g] - noise_floor
                for_fitting[g] = valid & (corrected_v > 0)
            else:
                # Nothing valid here for this gain, skip
                print(f'Insufficient data to fit slope for gain mode {g}, skipping and using nominal gain')
                continue
                
            # Write the median/variance/voltage data to an output table
            flat_table = Table([med_sig[g], var[g], voltage[g], exp_times[g], for_fitting[g]],
                                names=('Median','Variance','Voltage','Exptime','Valid'),
                                meta={'comments': f'Median and variance in ADU. Using {x_size}x{y_size} subframes. Bad pixels are {mask_str}.'})
            flat_table.write(os.path.join(output_dirname,f'ptc_data_{g.translate(str.maketrans("", "", " (-)"))}.ecsv'))
            
            try:
                # Fit while rejecting outliers and not over-weighting high signal data points
                model_fit[g], mask_ptc[g] = fit_or(model, med_sig[g][for_fitting[g]], corrected_v[for_fitting[g]], maxiter=100)

                #K_ADC(e−/DN) is determined from the slope of the shot noise curve (i.e. powerlaw amplitude)
                k = 1 / model_fit[g].amplitude.value
                if fit_or.fit_info['param_cov']:
                    k_err = np.abs(k * -1 * np.sqrt(fit_or.fit_info['param_cov'][0][0]) / model_fit[g].amplitude.value)
                else:
                    print('Error with calculating gain covariance')
                    k_err = 0

                det_gain[g] = [k, k_err]
            except:
                print(f'Problem fitting slope for gain mode {g}, skipping and using nominal gain')
    
    # Convert all properties to electron units using either measured gain or nominal gain
    gain_note = {}
    for g in gain_modes:
        gs = g.translate(str.maketrans("", "", " (-)")) # Simple gain sting for filenames
    
        if g in det_gain:
            use_gain[g] = det_gain[g][0]
            gain_comment = f'; Converted to electron units using measured gain {det_gain[g][0]:.3f} +/- {det_gain[g][1]:.3f} e-/ADU'
        else:
            use_gain[g] = nom_gain[g]
            gain_comment = f'; Converted to electron units using nominal (NOT measured) gain {nom_gain[g]} e-/ADU'
        
        if g in read_noise: read_noise_e[g] = read_noise[g] * use_gain[g]
        if g in dark_current: dark_current_e[g] = dark_current[g] * use_gain[g] / longdark_exp_time
        
        if bias_present:
            if g in med_bias_frames:
                # Create electron-unit frames
                bias_comment[g] += gain_comment
                noise_comment[g] += gain_comment
                med_bias_frames_e[g] = med_bias_frames[g] * use_gain[g]
                noise_map_e[g] = noise_map[g] * use_gain[g]
                
                # Plot individual bias/noise frames
                make_plot(med_bias_frames[g], med_bias_frames_e[g], f'Median bias frame - gain: {g}',
                          os.path.join(output_dirname,f'{output_prefix}_bias_frame_{gs}.png'))
                make_plot(noise_map[g], noise_map_e[g], f'Noise map - gain: {g}',
                          os.path.join(output_dirname,f'{output_prefix}_noise_map_{gs}.png'))
                
                # Define bad pixels as having noise above given threshold in high gain
                if (g == 'high') | (g == 'high (dual-gain)'):
                    noise_bad[g] = noise_map_e[g] >= noise_bad_thresh
                    noise_percent_bad[g] = np.sum(noise_bad[g]) / noise_map_e[g].size * 100
                    bad_pixel_map[noise_map_e[g] >= noise_bad_thresh] = 1
                    
        if longdark_present:
            if g in longdark_frames:
                # Create electron/s unit frames
                longdark_comment[g] += gain_comment+f'; Converted to rate using exposure time: {longdark_exp_time} s.'
                longdark_frames_e[g] = (longdark_frames[g] * use_gain[g]) / longdark_exp_time
                
                # Plot individual dark frames
                make_plot(longdark_frames[g], longdark_frames_e[g], f'Median long dark frame - gain: {g}',
                          os.path.join(output_dirname,f'{output_prefix}_longdark_frame_{gs}.png'), eunit='e-/s')
                
                # Define bad pixels as having dark current exceeding given threshold in high gain
                if (g == 'high') | (g == 'high (dual-gain)'):
                    dark_bad[g] = longdark_frames_e[g] > dark_bad_thresh
                    dark_percent_bad[g] = np.sum(dark_bad[g]) / longdark_frames_e[g].size * 100
                    bad_pixel_map[longdark_frames_e[g] > dark_bad_thresh] = 2
                    
        if flat_present:
            if g in med_sig:
                well_depth[g], well_depth_e[g] = np.nanmax(med_sig[g]), np.nanmax(med_sig[g]) * use_gain[g]
        
            if g in mid_flats:
                mid_flat_comment[g] += gain_comment
                mid_flats_e[g] = mid_flats[g] * use_gain[g]
                
                # Plot individual mid-range flat frames
                make_plot(mid_flats[g], mid_flats_e[g], f'Sample flat frame - gain: {g}',
                          os.path.join(output_dirname,f'{output_prefix}_flat_frame_{gs}.png'))
                
                # Define bad pixels as particularly low-response, having 5-sigma lower signal than mean pixel
                # TODO: find a better way to define bad pixels in this case?
                flat_bad_thresh[g] = np.mean(mid_flats_e[g]) - 5*np.std(mid_flats_e[g])
                flat_bad[g] = mid_flats_e[g] < flat_bad_thresh[g]
                flat_percent_bad[g] = np.sum(flat_bad[g]) / mid_flats_e[g].size * 100
                bad_pixel_map[mid_flats_e[g] < flat_bad_thresh[g]] = 3
                
    if opdark_present:
        long_high_opdark_e, long_low_opdark_e, short_high_opdark_e, short_low_opdark_e = {}, {}, {}, {}
        for mode in long_high_opdark:
            # Convert operating darks to e-/s units
            long_high_opdark_e[mode] = long_high_opdark[mode] * use_gain['high (dual-gain)'] / oplong[mode]
            long_low_opdark_e[mode] = long_low_opdark[mode] * use_gain['low (dual-gain)'] / oplong[mode]
            short_high_opdark_e[mode] = short_high_opdark[mode] * use_gain['high (dual-gain)'] / opshort[mode]
            short_low_opdark_e[mode] = short_low_opdark[mode] * use_gain['low (dual-gain)'] / opshort[mode]
    
    if persist_present:
        persist_sat_median_e = persist_sat_median * use_gain['high']
        persist_dark_frames_e = persist_dark_frames * use_gain['high']

    # Write all these FITS files again in electron (/s) units
    if bias_present:
        bias_e_outpath = write_fits_image(med_bias_frames_e, 'bias frame', bias_comment,
                                          os.path.join(output_dirname,f'{output_prefix}_bias_frames_e.fits'))
        noise_e_outpath = write_fits_image(noise_map_e, 'noise map', noise_comment,
                                           os.path.join(output_dirname,f'{output_prefix}_noise_maps_e.fits'))
    if longdark_present:
        longdark_e_outpath = write_fits_image(longdark_frames_e, 'long dark frame', longdark_comment,
                                              os.path.join(output_dirname,f'{output_prefix}_longdark_frames_e.fits'))
    if flat_present:
        flat_e_outpath = write_fits_image(mid_flats_e, 'sample flat frame', mid_flat_comment,
                                          os.path.join(output_dirname,f'{output_prefix}_flat_frames_e.fits'))
    
    
    # TODO: Add all the electron unit frame stats to some sort of output table
    
    ###############################
    # PDF report generation section
    # - Produces PDF summary report
    ###############################
    
    # Plot settings
    gain_color = {'high (dual-gain)': 'tab:blue', 'low (dual-gain)': 'tab:orange', 'high': 'tab:green', 'low': 'tab:red'}
    gain_line_color = {'high (dual-gain)': 'darkblue', 'low (dual-gain)': 'brown', 'high': 'darkgreen', 'low': 'darkred'}
    
    # Initialize report document
    #doc_name = f'analysis_report_test.pdf'
    with PdfPages(os.path.join(output_dirname,doc_name)) as pdf:
    
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    
        # Empty axes for first page
        fig = plt.figure(figsize=[8.5,11],dpi=250)
        plt.axis('off')
        
        # Temperature
        if exp.temperature > 0: temp = f'{exp.temperature:2f} K (measured)'
        else: temp = 'TEMPERATURE DATA MISSING (default is 140K)'
        
        # Device and test summary text
        summary_text = f'Cosmetic report for DeviceID: {detid}\n'
        summary_text += f'Device size: {exp.dev_size[0]} x {exp.dev_size[1]}\n\n'
        summary_text += f'Standard analysis exposures taken {datestring} using {camera} camera\n'
        summary_text += f'with firmware: {exp.firmware}\n'
        summary_text += f'at temperature {temp}\n'
        if ledwave: summary_text += f'using LED: {ledwave} nm\n'
        summary_text += '\n'
        summary_text += f'Report generated {now}; cmost-analysis git hash {git_hash} \n\n'
        if 'graycode' in kwargs: summary_text += f'Gray decoding applied \n\n'
        if 'subframe' in kwargs: summary_text += f'Results for subframe (x1,x2,y1,y2): {subframe} \n\n'
        
        summary_text += 'Contents:\n'
        if singleframe_present: summary_text += '- Single readout frames\n'
        if noise_present: summary_text += '- Noise spectrum\n'
        if bias_present: summary_text += '- Bias frames\n'
        if longdark_present: summary_text += '- Long dark frames\n'
        if opdark_present: summary_text += '- Standard operating dark frames\n'
        if persist_present: summary_text += '- Persistence test frames\n'
        if flat_present: summary_text += f'- Flat fields vs exposure time\n'
        
        # Include any notes from the analysis_notes file
        if notes_present:
            summary_text += '\n\nNotes:'
            if len(notes_lines[2:]) < 20:
                summary_text += ''.join(notes_lines[3:])
            else:
                summary_text += ''.join(notes_lines[3:21])+'...'
                summary_text += '\nNotes continued in analysis_notes.txt'
        
        plt.text(0,1,summary_text,verticalalignment='top',wrap=True)
        fig.text(0.96, 0.02, pdf.get_pagecount()+1)
        pdf.savefig()
        plt.close()
        
        # So-what summary pages at the top
        if bias_present or longdark_present or flat_present:
            # Bad pixel map page
            fig = plt.figure(figsize=[8.5,11],dpi=250)
            plt.suptitle('Bad pixel map')
            ax1 = plt.subplot2grid((2,1), (0,0))
            plt.sca(ax1)
            bad_color_map = matplotlib.colors.ListedColormap(['white', 'red', 'blue', 'limegreen'])
            plt.imshow(bad_pixel_map+0.5, cmap=bad_color_map, vmin=0, vmax=4)
            plt.colorbar(orientation='horizontal', label='Bad pixel type', ticks=[0.5,1.5,2.5,3.5],
                         format=mticker.FixedFormatter(['Good', 'High Noise', 'High Dark', 'Low Sensitivity']))
            
            # Get combined numbers and percentages of bad pixels
            summary_text = ''
            
            if bias_present:
                total_noise_bad = np.sum(np.any(np.stack(list(noise_bad.values())),axis=0))
                total_noise_percent_bad = total_noise_bad / bad_pixel_map.size * 100
                summary_text += f'High noise pixels (>{noise_bad_thresh} e-): {total_noise_bad} ({total_noise_percent_bad:.2f} %)\n'
            if longdark_present:
                total_dark_bad = np.sum(np.any(np.stack(list(dark_bad.values())),axis=0))
                total_dark_percent_bad = total_dark_bad / bad_pixel_map.size * 100
                summary_text += f'High dark current pixels (>{dark_bad_thresh} e-/s): {total_dark_bad} ({total_dark_percent_bad:.2f} %)\n'
            if flat_present:
                total_flat_bad = np.sum(np.any(np.stack(list(flat_bad.values())),axis=0))
                total_flat_percent_bad = total_flat_bad / bad_pixel_map.size * 100
                summary_text += f'Low sensitivity current pixels (<5-sigma below mean): {total_flat_bad} ({total_flat_percent_bad:.2f} %)\n'
            total_percentage_bad = np.sum((bad_pixel_map > 0)) / bad_pixel_map.size * 100

            summary_text += f'Total number of bad pixels: {int(np.sum(bad_pixel_map))} ({total_percentage_bad:.2f} %)'
            plt.text(-0.5, -1, summary_text, transform=ax1.transAxes, verticalalignment='top')
            
            pdf.savefig()
            plt.close()
        
            # Final summary page
            fig = plt.figure(figsize=[8.5,11],dpi=250)
            plt.axis('off')
            
            plt.xlim(0,10)
            plt.ylim(0,10)
            
            # Header
            c1, c2, c3, c4, c5 = 0, 3.9, 5.8, 8.0, 10.0
            plt.text(c1, 9.5, 'Gain mode', weight='bold', ha='left')
            plt.text(c2, 9.5, 'Gain', weight='bold', ha='right')
            plt.text(c3, 9.5, 'Read Noise', weight='bold', ha='right')
            plt.text(c4, 9.5, 'Dark Current', weight='bold', ha='right')
            plt.text(c5, 9.5, 'Well Depth', weight='bold', ha='right')
            plt.text(c2, 9.2, '(e-/ADU)', weight='bold', ha='right')
            plt.text(c3, 9.2, '(e-)', weight='bold', ha='right')
            plt.text(c4, 9.2, '(99%; me-/s)', weight='bold', ha='right')
            plt.text(c5, 9.2, '(e-)', weight='bold', ha='right')
            
            # Results
            for i, g in enumerate(gain_modes):
                plt.text(c1, 8.5-i*0.5, g)
                if g in det_gain: plt.text(c2, 8.5-i*0.5, f'{det_gain[g][0]:.3f}±{det_gain[g][1]:.3f}', ha='right')
                else: plt.text(c2, 8.5-i*0.5, f'{nom_gain[g]:.1f} (nom.)', ha='right')
                if g in read_noise: plt.text(c3, 8.5-i*0.5, f'{read_noise_e[g]:.2f}', ha='right')
                if g in dark_current: plt.text(c4, 8.5-i*0.5, f'{dark_current_e[g]*1000:.2f}', ha='right')
                if g in well_depth: plt.text(c5, 8.5-i*0.5, f'{int(well_depth_e[g])}', ha='right')
            
            fig.text(0.96, 0.02, pdf.get_pagecount()+1)
            pdf.savefig()
            plt.close()
        
        # Bias plots
        if bias_present:
            # Now create bias and noise map summary pages
            for g in med_bias_frames_e:
                # Plot bias frame
                mmin, mmax = min(med_bias_frames_e[g].flatten()), max(med_bias_frames_e[g].flatten())
                
                summary_text = f'Gain mode: {g}\n'
                summary_text += '\n'.join(bias_comment[g].split('; '))+'\n'
                summary_text += f'Min median pixel value: {mmin:.1f} e-; max median pixel value: {mmax:.1f} e- \n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, med_bias_frames_e[g], f'Bias frames - gain: {g}', summary_text, precision=use_gain[g])
                
                # Also plot read noise map and histogram
                nmin, nmax = min(noise_map_e[g].flatten()), max(noise_map_e[g].flatten())
                
                summary_text = f'Gain mode: {g}\n'
                summary_text += '\n'.join(bias_comment[g].split('; '))+'\n'
                summary_text += f'Min noise value: {nmin:.1f} e-; max noise value: {nmax:.1f} e- \n'
                summary_text += f'Read noise (RMS): {read_noise_e[g]:.2f} e-\n'
                if (g == 'high') | (g == 'high (dual-gain)'):
                    summary_text += f'Percentage above {noise_bad_thresh} e-: {noise_percent_bad[g]:.2f} %\n'
                    hist_page(pdf, noise_map_e[g], f'Noise map - gain: {g}', summary_text, precision=0.1,
                              unit='Read Noise (e-)', vlines=[noise_bad_thresh])
                else:
                    hist_page(pdf, noise_map_e[g], f'Noise map - gain: {g}', summary_text, precision=0.1, unit='Read Noise (e-)')
            
            # Read noise summary page
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Read noise summary')
            
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = ax1.twinx()
            
            # Read noise summary plot
            plt.sca(ax1)
            summary_text = ''
            for g in noise_map_e:
                bins = np.arange(min(noise_map_e[g].flatten()),max(noise_map_e[g].flatten()),0.1)
                data_hist, bin_edges = np.histogram(noise_map_e[g],bins=bins)
                p = ax1.stairs(data_hist / noise_map_e[g].size, edges=bin_edges, label=g)
                ax2.plot(bins[:-1], np.cumsum(data_hist / noise_map_e[g].size), color=p.get_edgecolor(), ls='--')
                summary_text += f'RMS read noise, {g}: {read_noise_e[g]:.2f} e-\n'
            plt.axvline(3,ls='--',color='grey',label='3e- Requirement')
            plt.xlabel('Read Noise (e-)')
            ax1.set_ylabel('Fraction of Pixels')
            min_y = 1 / (noise_map_e[g].size*2)
            ax1.set_ylim(min_y,1)
            ax2.set_ylabel('Cumulative Fraction of Pixels')
            ax2.set_ylim(0,1)
            ax1.loglog()
            plt.legend(fontsize=10,loc=1)
            
            plt.text(0.0, -0.3, summary_text, transform=ax1.transAxes, verticalalignment='top')
            
            plt.tight_layout()
            fig.text(0.96, 0.02, pdf.get_pagecount()+1)
            pdf.savefig()
            plt.close()
            
        # Noise spectrum
        if noise_present:
            fig, axs = plt.subplots(figsize=[8.5,11],dpi=300,nrows=4,gridspec_kw={'hspace': 0})
            plt.suptitle(f'Noise Spectra')
            labels = [f'Channel {c}' for c in range(n_channels)]
            # We expect 3 frames from a standard exposure set but be flexible
            for i in range(np.minimum(len(noise_spectrum),3)):
                axs[i].loglog()
                axs[i].plot(noise_freq[i], noise_spectrum[i], label=labels)
                if i < len(noise_spectrum)-1: axs[i].get_xaxis().set_visible(False)
                axs[i].set_xlabel('Frequency (Hz)')
                if i == 0:
                    axs[i].set_ylabel(r'Noise Density (V / Hz$^2$)')
                    axs[i].legend(fontsize=10,ncols=4)
                axs[i].text(0.05,0.1,f'Pixel: {noise_pix[i]}',transform=axs[i].transAxes)
            for a in axs[np.minimum(len(noise_spectrum),3):]:
                a.axis('off')
            pdf.savefig()
            plt.close()

        # Long darks
        if longdark_present:
            for g in longdark_frames_e:
                # Plot long dark frames
                frame = longdark_frames_e[g]
                dmin, dmax, dmedian, dmean = min(frame.flatten()), max(frame.flatten()), np.median(frame), np.mean(frame)
                
                if 'subframe' not in kwargs:
                    # Boxcar smooth for plotting and histogram, if not looking at a subframe
                    smoothed_dark = convolve(frame, Box2DKernel(5))
                    smoothed_txt = 'Frame is boxcar-smoothed with width=5 pixels\n'
                    plot_dark = smoothed_dark
                else:
                    smoothed_txt = ''
                    plot_dark = frame
                
                # Create a standard histogram page and save it to the pdf
                summary_text = f'Frame: {g} long dark\n'
                summary_text += '\n'.join(longdark_comment[g].split('; '))+f'\n{smoothed_txt}'
                summary_text += f'Min median pixel value: {dmin:.4f} e-/s; max median pixel value: {dmax:.4f} e-/s \n'
                summary_text += f'Median pixel value: {dmedian:.4f} e-/s; mean pixel value: {dmean:.4f} e-/s\n'
                
                # Mark the read noise level as well as the bad threshold for high-gain
                read_noise_level = read_noise_e[g] / longdark_exp_time

                prec = (use_gain[g] / longdark_exp_time) * 5
                if (g == 'high') | (g == 'high (dual-gain)'):
                    summary_text += f'Percentage above {dark_bad_thresh} e-/s: {dark_percent_bad[g]:.2f} %; 99% percentile: {dark_current_e[g]:.4f} e-/s\n'
                    hist_page(pdf, plot_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec,
                              contours=[dark_bad_thresh], vlines=[dark_bad_thresh, read_noise_level])
                else:
                    hist_page(pdf, plot_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec, vlines=[read_noise_level])

        # Persistence test
        if persist_present:
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Persistence test')
            
            # Output a frame image and histograms
            ax1 = plt.subplot2grid((4,2), (0,0))
            darkax = [plt.subplot2grid((4,2), (0,1)), plt.subplot2grid((4,2), (1,0)), plt.subplot2grid((4,2), (1,1))]
            ax5 = plt.subplot2grid((4,1), (2,0))
            
            # Median saturated frame
            plt.sca(ax1)
            plt.title('Saturated frame')
            plt.imshow(persist_sat_median_e, vmin=np.percentile(persist_sat_median_e,0.03), vmax=np.percentile(persist_sat_median_e,99.7))
            plt.colorbar(label='e-', shrink=0.9)
            
            # First three dark frames
            for i, ax in enumerate(darkax):
                plt.sca(ax)
                plt.title(f'Post-saturated dark {i+1}')
                imdata = persist_dark_frames_e[i]
                # Set scaling to match the first frame in sequence
                if i == 0: vmin, vmax = np.percentile(imdata,0.03), np.percentile(imdata,99.7)
                plt.imshow(imdata, vmin=vmin, vmax=vmax)
                plt.colorbar(label='e-', shrink=0.9)
            
            # Plot of dark median over time
            dark_medians = np.nanmedian(persist_dark_frames_e, axis=(1,2))
            
            plt.sca(ax5)
            plt.plot(persist_times, dark_medians)
            plt.xlabel(f'Time ({persist_tu})')
            plt.ylabel('Mean Signal (e-)')
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()

        # Flats, linearity and PTCs
        if flat_present:
            # Plot mid-range flats
            for g in mid_flats_e:
                flat = mid_flats_e[g]
                mmin, mmax, mmedian, mmean = min(flat.flatten()), max(flat.flatten()), np.median(flat), np.mean(flat)
                
                if flatdark_present: summary_text = f'Gain mode: {g}, equal-length dark subtracted\n'
                else: summary_text = f'Gain mode: {g}{bias_note}\n'
                summary_text += '\n'.join(mid_flat_comment[g].split('; '))+'\n'
                #summary_text += f'Illuminated {mid_flat_times[g]:.1f}s exposure; LED at {mid_flat_voltages[g]} V\n'
                summary_text += f'Min pixel value: {mmin:.1f} e-; max pixel value: {mmax:.1f} e- \n'
                summary_text += f'Median pixel value: {mmedian:.1f} e-; mean pixel value: {mmean:.1f} e-\n'
                summary_text += f'Percentage below {flat_bad_thresh[g]:.1f} e-/s (5-sigma below mean): {flat_percent_bad[g]:.2f} %\n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, flat, f'Mid-range flat frame - gain: {g}', summary_text, precision=min([50,mmax/50]))

            # Linearity and PTC plots
            
            # Linearity curve
            for g in med_sig:
                # Each gain mode gets its own page
                fig = plt.figure(figsize=[8.5,11],dpi=300)
                plt.suptitle(f'Linearity plots')
                gs_top = plt.GridSpec(2, 1, top=0.9)
                ax1 = fig.add_subplot(gs_top[0,:])

                # A subplot for each voltage (we expect 4)
                gs_bottom = gs_base = plt.GridSpec(9, 1, hspace=0)
                ax2 = fig.add_subplot(gs_base[5,:])
                ax3 = fig.add_subplot(gs_base[6,:], sharex = ax2)
                ax4 = fig.add_subplot(gs_base[7,:], sharex = ax2)
                ax5 = fig.add_subplot(gs_base[8,:], sharex = ax2)
                vol_subplots = [ax5,ax4,ax3,ax2]
                plt.subplots_adjust(hspace=0)
                
                # Plot the signal vs exposure time for each voltage
                led_vals = np.unique(voltage[g])
                alpha = np.maximum(1/(2*n_subframes), 0.05)
                for i, vol in enumerate(led_vals):
                    plt.sca(ax1)
                    means = np.array([])
                    exposure_times = np.unique(exp_times[g])
                    this_voltage = voltage[g] == vol
                    
                    # Get mean over detector subframes
                    for t in exposure_times:
                        this_voltage_time = this_voltage & (exp_times[g] == t)
                        means = np.append(means, np.mean(med_sig[g][this_voltage_time]))
                    
                    # Plot signal and mean signal
                    plt.scatter(exp_times[g][this_voltage], med_sig[g][this_voltage], marker='x', label=f'{g}', color=gain_color[g], alpha=alpha)
                    plt.scatter(exposure_times, means, marker='x', label=f'{g}', color=gain_color[g])
                    
                    # Fit line to the set of means above 10 and less than the rough saturation level
                    for_fitting_lin = (means > 10) & (means < (max(med_sig[g]) - 5000))
                    if np.sum(for_fitting_lin) > 1:
                        model_fit_lin, mask_lin = fit_or(model, exposure_times[for_fitting_lin], means[for_fitting_lin], maxiter=100)
                        plt.plot(np.logspace(-0.3,2.4),model_fit_lin(np.logspace(-0.3,2.4)), ls='--', color=gain_line_color[g])
                        if model_fit_lin(0.8) > 0.5: plt.text(1.3,model_fit_lin(0.8),f'{vol} V')
                        else: plt.text(150,model_fit_lin(250),f'{vol} V')
                    
                        # Plot residuals from that fit altogether for each voltage as % from fit
                        plt.sca(vol_subplots[i])
                        plt.plot([0.5,30000],[0,0],ls='--',color='gray')
                        resid = (med_sig[g][this_voltage] - model_fit_lin(exp_times[g][this_voltage])) / model_fit_lin(exp_times[g][this_voltage]) * 100
                        resid_means = (means - model_fit_lin(exposure_times)) / model_fit_lin(exposure_times) * 100
                        plt.scatter(med_sig[g][this_voltage], resid, marker='x', label=f'{g}', color=gain_color[g], alpha=alpha)
                        plt.scatter(means, resid_means, marker='x', label=f'{g}', color=gain_color[g])
                        plt.text(0.02,0.1,f'{vol} V',transform=vol_subplots[i].transAxes)
                        plt.semilogx()
                        plt.xlim(0.5,30000)
                    else:
                        plt.text(1.3,max(med_sig[g][this_voltage]),f'{vol} V')
                        plt.sca(vol_subplots[i])
                        plt.plot([0.5,30000],[0,0],ls='--',color='gray')
                        plt.text(0.02,0.1,f'{vol} V',transform=vol_subplots[i].transAxes)
                        plt.semilogx()
                        plt.xlim(0.5,30000)
            
                plt.sca(ax1)
                plt.xlabel('Exposure Time (s)')
                plt.ylabel('Median Signal (ADU)')
                plt.xlim(1.1,250)
                plt.ylim(0.5,30000)
                plt.loglog()
                
                # Avoid duplicate legend labels
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), fontsize=10,ncol=4,loc=8,bbox_to_anchor=(0.5, 1.0))
                
                ax3.set_ylabel('% deviation from linear fit')
                ax5.set_xlabel('Signal (ADU)')
                
                fig.text(0.96, 0.02, pdf.get_pagecount()+1)
                pdf.savefig()
                plt.close()
            
            # PTCs
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Photon Transfer Curve')
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))

            for g in med_sig:
                # Fit PTC for read noise level and gain
                if (g == 'high (dual-gain)') | (g == 'low (dual-gain)'): plt.sca(ax1)
                else: plt.sca(ax2)
                
                # Is there a gain measurement here?
                if g in det_gain:
                    # Plot points used for fitting and points filtered out - mask filters out
                    plt.scatter(med_sig[g][for_fitting[g]][mask_ptc[g]], var[g][for_fitting[g]][mask_ptc[g]],
                                marker='x', color=gain_color[g], alpha=alpha)
                    plt.scatter(med_sig[g][~for_fitting[g]], var[g][~for_fitting[g]],
                                marker='x', color=gain_color[g], alpha=alpha)
                    plt.scatter(med_sig[g][for_fitting[g]][~mask_ptc[g]], var[g][for_fitting[g]][~mask_ptc[g]],
                                marker='x', color=gain_color[g])

                    # Plot fit
                    plt.plot(np.logspace(1,4.3), model_fit[g](np.logspace(1,4.3)), ls='--', color=gain_line_color[g],
                             label=f'{g} gain: {det_gain[g][0]:.3f} ± {det_gain[g][1]:.3f} e-/ADU')
                    
                else:
                    # Just plot the data points without fit
                    plt.scatter(med_sig[g], var[g], marker='x', color=gain_color[g], alpha=alpha, label=f'{g} gain: no gain measurement')
                    
                # Plot noise and well depth
                if g in read_noise: noise_floor = read_noise[g]**2
                else: noise_floor = nom_read_noise[g]**2
                plt.plot([1,2e4],[noise_floor, noise_floor], ls='--', color=gain_line_color[g], alpha=0.5)
                plt.plot([max(med_sig[g]),max(med_sig[g])], [1,max(var[g])], ls='--', color=gain_line_color[g], alpha=0.5)
            
            plt.sca(ax1)
            plt.legend(loc=2)
            plt.xlabel('Median Signal (ADU)')
            plt.ylabel('Variance (ADU^2)')
            plt.loglog()
            plt.text(1, 0.8, 'Read Noise', color='grey')
            
            plt.sca(ax2)
            plt.legend(loc=2)
            plt.xlabel('Median Signal (ADU)')
            plt.ylabel('Variance (ADU^2)')
            plt.loglog()
            plt.text(1, 0.8, 'Read Noise', color='grey')

            plt.tight_layout()
            fig.text(0.96, 0.02, pdf.get_pagecount()+1)
            plt.savefig(os.path.join(output_dirname,f'{output_prefix}_ptc.png'))
            pdf.savefig()
            plt.close()
            
        # TODO: something for guiding rows
        
        # Standard operating darks
        if opdark_present:
            for mode in long_high_opdark:
                # For each resulting median frame, print a page of plots
                n_frames = len(long_high_opdark[mode])
                frames = [np.median(short_low_opdark_e[mode],axis=0), np.median(short_high_opdark_e[mode],axis=0),
                          np.median(long_low_opdark_e[mode],axis=0), np.median(long_high_opdark_e[mode],axis=0)]
                names = ['short low-gain','short high-gain','long low-gain','long high-gain']
                precisions = [use_gain['low (dual-gain)']/opshort[mode], use_gain['high (dual-gain)']/opshort[mode],
                              use_gain['low (dual-gain)']/oplong[mode], use_gain['high (dual-gain)']/oplong[mode]]
                times = [opshort[mode], opshort[mode], oplong[mode], oplong[mode]]
                
                for j, med_dark in enumerate(frames):
                    mmin, mmax, mmedian, mmean = min(med_dark.flatten()), max(med_dark.flatten()), np.median(med_dark), np.mean(med_dark)
                    
                    summary_text = f'Frame: {names[j]}, no bias subtraction\n'
                    summary_text += f'Median of {n_frame} x {times[j]}s exposures\n'
                    summary_text += f'Min median pixel value: {mmin:.2f} e-/s; max median pixel value: {mmax:.2f} e-/s \n'
                    summary_text += f'Median pixel value: {mmedian:.2f} e-/s; mean pixel value: {mmean:.2f} e-/s'
                    
                    # Create a standard histogram page and save it to the pdf
                    hist_page(pdf, med_dark, f'Dark frames - {mode} mode - {names[j]} frame',
                              summary_text, precision=precisions[j], unit='e-/s')
                
                # Also make a page of mean values over time
                fig = plt.figure(figsize=[8.5,11],dpi=250)
                plt.suptitle(f'Dark frames - {mode} mode - mean over time')
                
                ax1 = plt.subplot2grid((4,1), (0,0))
                ax2 = plt.subplot2grid((4,1), (1,0))
                ax3 = plt.subplot2grid((4,1), (2,0))
                ax4 = plt.subplot2grid((4,1), (3,0))
                
                # Mean short-frame dark value over time (whole frame + every column)
                sl_mean = np.mean(short_low_opdark[mode], axis=(1,2)) * use_gain['low (dual-gain)']
                sl_std = np.std(short_low_opdark[mode], axis=(1,2)) * use_gain['low (dual-gain)']
                sh_mean = np.mean(short_high_opdark[mode], axis=(1,2)) * use_gain['high (dual-gain)']
                sh_std = np.std(short_high_opdark[mode], axis=(1,2)) * use_gain['high (dual-gain)']
                ll_mean = np.mean(long_low_opdark[mode], axis=(1,2)) * use_gain['low (dual-gain)']
                ll_std = np.std(long_low_opdark[mode], axis=(1,2)) * use_gain['low (dual-gain)']
                lh_mean = np.mean(long_high_opdark[mode], axis=(1,2)) * use_gain['high (dual-gain)']
                lh_std = np.std(long_high_opdark[mode], axis=(1,2)) * use_gain['high (dual-gain)']
                
                if 'subframe' not in kwargs:
                    # Also show a column-by-column breakdown if not looking at a subframe
                    sl_col_mean, sh_col_mean, ll_col_mean, lh_col_mean = [], [], [], []
                    for i in range(n_channels):
                        sl_col_mean.append(np.mean(short_low_opdark[mode][:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['low (dual-gain)'])
                        sh_col_mean.append(np.mean(short_high_opdark[mode][:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['high (dual-gain)'])
                        ll_col_mean.append(np.mean(long_low_opdark[mode][:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['low (dual-gain)'])
                        lh_col_mean.append(np.mean(long_high_opdark[mode][:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['high (dual-gain)'])
                    frame_label = 'Whole detector'
                else:
                    frame_label = f'Subframe: {kwargs["subframe"]}'

                plt.sca(ax1)
                plt.title(f'{names[0]}')
                plt.plot(s_starts_opdark[mode],sl_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts_opdark[mode],sl_mean-sl_std,sl_mean+sl_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): p = plt.plot(s_starts_opdark[mode],sl_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                plt.legend(fontsize=10,ncol=6,loc=9)
                
                plt.sca(ax2)
                plt.title(f'{names[1]}')
                plt.plot(s_starts_opdark[mode],sh_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts_opdark[mode],sh_mean-sh_std,sh_mean+sh_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(s_starts_opdark[mode],sh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                # Mean long-frame dark value over time
                plt.sca(ax3)
                plt.title(f'{names[2]}')
                plt.plot(l_starts_opdark[mode],ll_mean,'x-',color='k',label=frame_label)
                plt.fill_between(l_starts_opdark[mode],ll_mean-ll_std,ll_mean+ll_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(l_starts_opdark[mode],ll_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                plt.sca(ax4)
                plt.title(f'{names[3]}')
                plt.plot(l_starts_opdark[mode],lh_mean,'x-',color='k',label=frame_label)
                plt.fill_between(l_starts_opdark[mode],lh_mean-lh_std,lh_mean+lh_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(l_starts_opdark[mode],lh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.xlabel('Time (s)')
                plt.ylabel('e-')
                
                plt.tight_layout()
                fig.text(0.96, 0.02, pdf.get_pagecount()+1)
                pdf.savefig()
                plt.close()


def hist_page(pdf, data, title, summary_text, unit='e-', precision=1, contours=False, vlines=False):
    '''
    Default style of page showing a figure of the image, histograms, and summary text
    '''
    dmin, dmax = min(data.flatten()), max(data.flatten())
    data_range = int( (dmax - dmin) / precision ) + 1

    fig = plt.figure(figsize=[8.5,11],dpi=250)
    plt.suptitle(title)
    
    data_hist, bin_edges = np.histogram(data,bins=min([100,data_range]))
    data_hist2, bin_edges2 = np.histogram(data,bins=data_range) # i.e. binsize=precision

    # Output a frame image and histograms
    ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((5,2), (2,0), rowspan=2)
    ax3 = plt.subplot2grid((5,2), (2,1), rowspan=2)
                    
    plt.sca(ax1)
    # Scale to median +/- 3-sigma to have outliers pop out
    plt.imshow(data, vmin=np.percentile(data,0.03), vmax=np.percentile(data,99.7))
    plt.colorbar(label=unit, shrink=0.9)
    if contours:
        plt.contour(gaussian_filter(data,2), levels=contours, colors='r', linewidths=0.5)
    
    plt.sca(ax2)
    plt.stairs(data_hist,edges=bin_edges)
    plt.xlabel(unit)
    plt.ylabel('Pixel Count')
    plt.text(0.55,0.95,'Whole range',transform=ax2.transAxes)
    plt.semilogy()
    
    plt.sca(ax3)
    plt.stairs(data_hist2,edges=bin_edges2)
    plt.xlabel(unit)
    plt.xlim(np.percentile(data,[1.e-2,100-1.e-2]))
    plt.text(0.35,0.95,'0.01 - 99.99 percentile',transform=ax3.transAxes)
    if vlines:
        for vl in vlines:
            plt.axvline(vl,ls='--',color='grey')
    plt.semilogy()
                    
    plt.text(0.0, -0.3, summary_text, transform=ax2.transAxes, verticalalignment='top')

    plt.subplots_adjust(hspace=0.25)
    
    fig.text(0.96, 0.02, pdf.get_pagecount()+1)
    pdf.savefig()
    plt.close()


def make_plot(adudata, edata, title, output_path, eunit='e-'):
    '''
        Outputs a PNG of a given frame with axes in ADU and electron units
    '''
    fig, ax = plt.subplots(figsize=[8,6],dpi=250)
    plt.title(title)
    # Scale to median +/- 3-sigma to have outliers pop out
    # Because ADU and e data are scaled, the images themselves will be identical but the colorbars different
    aduplot = ax.imshow(adudata, vmin=np.percentile(adudata,0.03), vmax=np.percentile(adudata,99.7))
    eplot = ax.imshow(edata, vmin=np.percentile(edata,0.03), vmax=np.percentile(edata,99.7))
    bar1 = plt.colorbar(aduplot, label='ADU', shrink=0.85)
    bar2 = plt.colorbar(eplot, label=eunit, shrink=0.85)
    plt.savefig(output_path)
    plt.close()


def write_fits_image(frames, frame_type, comment, output_file):
    '''
        Writes a FITS image file, given a gain-indexed dictionary of frames
    '''
    # Initialize FITS image file
    hdus = [fits.PrimaryHDU()]
    
    # Loop through dictionary of frames and create image HDU for each
    for g in frames:
        imhdu = fits.ImageHDU(data=frames[g], name=f'{g} gain {frame_type}')
        imhdu.header['GAIN'] = g
        imhdu.header['TYPE'] = frame_type
        imhdu.header['COMMENT'] = comment[g]
        hdus.append(imhdu)
    hdulist = fits.HDUList(hdus=hdus)
    
    # Write to the provided output file
    hdulist.writeto(output_file)
    
    return output_file


if __name__ == '__main__':
    '''
    Usage:
    python [-g] [-s x1,x2,y1,y2] cmost_analysis.py DIRECTORY
    '''
    import argparse
    
    def int_tuple(arg):
        return tuple(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to data directory")
    parser.add_argument("-g", help="apply graycode descrambling", action="store_true")
    parser.add_argument("-s", "--subframe", type=int_tuple, help="subframe definition in form x1,x2,y1,y2")
    args = parser.parse_args()

    kwargs = {}
    if args.g == True:
        kwargs['graycode'] = True
    if args.subframe is not None:
        subframe = args.subframe
        assert len(subframe) == 4, 'Invalid subframe (must be in form x1,x2,y1,y2)'
        kwargs['subframe'] = subframe

    standard_analysis_products(args.directory,**kwargs)

