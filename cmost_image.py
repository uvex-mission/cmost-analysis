''' Functions relating to opening and manipulating CMOST images'''
import os
import numpy as np
from astropy.io import fits
from datetime import datetime

'''
    Function to open a CMOST image file
    Returns header data and useable frames
'''
def open_cmost_file(filename):
    
    cmost_file = fits.open(filename)
    cmost_hdr = cmost_file[0].header
    
    # DATE and READOUTM are set automatically by the camera control
    readout_mode = cmost_hdr['READOUTM']
    date = cmost_hdr['DATE']
    
    # Other headers are set by the user, so cope if they are not set
    if 'EXPTIME' in cmost_hdr.keys():
        exp_time = cmost_hdr['EXPTIME']
    else:
        exp_time = '-1'
    if 'LED' in cmost_hdr.keys():
        led_voltage = cmost_hdr['LED']
    else:
        led_voltage = '-1'
    if 'TEMP' in cmost_hdr.keys():
        temp = cmost_hdr['TEMP']
    else:
        temp = '-1'

    # Create an array of useable frames
    frames = []
    for i, ex in enumerate(cmost_file):
        # Ignore 0th extension and initial image (data is meaningless)
        if i > 1:
            # Frame data is in uint16 by default, open in int32
            frames.append(np.array(cmost_file[i].data, dtype=np.int32))
    frames = np.array(frames)

    cmost_file.close()
    
    return frames, readout_mode, date, exp_time, led_voltage, temp

'''
    Function to get the image frame from correlated double sampling input
'''
def get_corrected_level(cmost_frame):
    
    # Divide frame into signal level and reset level columns
    signal_xindices = np.array([(np.arange(256)+i*512) for i in range(16)]).flatten()
    reset_xindices = np.array([(np.arange(256)+i*512+256) for i in range(16)]).flatten()
    
    signal_level = cmost_frame[:,signal_xindices]
    reset_level = cmost_frame[:,reset_xindices]
    
    corrected_level = signal_level - reset_level
    
    return corrected_level

'''
    Load multiple files from wherever you're keeping data according to file prefix
    e.g. file_prefix = 'cmost001'
    Return corrected frames, header data, and a sample signal & rms
'''
def get_file_list(cmost_dir,file_prefix='cmost',debug=False):

    # Define files to load
    files = np.array(os.listdir(cmost_dir))
    if file_prefix != '':
        filefilter = [f[0:len(file_prefix)] == file_prefix for f in files]
        files = files[filefilter]
    num_files = len(files)

    # Populate a list of file properties
    all_frames, ro_modes, dates, exp_times, voltages, temps = [], [], [], [], [], []
    signals, rms = np.zeros(num_files), np.zeros(num_files)
    for i, cmost_file in enumerate(files):
        # Open file and get header info
        frames, ro_mode, date, exp, led_v, temp = open_cmost_file('{}/{}'.format(cmost_dir,cmost_file))
        ro_modes.append(ro_mode)
        dates.append(datetime.fromisoformat(date))
        exp_times.append(exp)
        voltages.append(led_v)
        temps.append(temp)
        
        # Get corrected frames
        corr_frames = []
        for f in frames:
            corrected_frame = get_corrected_level(f)
            corr_frames.append(corrected_frame)
        all_frames.append(np.array(corr_frames))
        
        # Define an illuminated 100x100 window and get the mean signal from the first frame
        frame1 = corr_frames[0]
        signals[i] = np.mean(frame1[950:1050,2950:3050])
        
        # If a second frame is available, also calculate the rms
        if len(corr_frames) > 1:
            frame2 = corr_frames[1]
            
            # Get the rms of this individual window
            frame_diff = frame2 - frame1 # Removes pixel-to-pixel noise
            variance = np.mean(frame_diff[950:1050,2950:3050]**2) / 2
            rms[i] = np.sqrt(variance)
        else:
            rms[i] = -1
        
        if debug:
            print(cmost_file, ro_mode, date, exp, led_v, temp, signals[i], rms[i])
                
    all_frames = np.array(all_frames)
    ro_modes = np.array(ro_modes)
    dates = np.array(dates)
    exp_times = np.array(exp_times)
    voltages = np.array(voltages)
    temps = np.array(temps)

    return all_frames, ro_modes, dates, exp_times, voltages, temps, signals, rms
