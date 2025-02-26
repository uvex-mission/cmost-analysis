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
from cmost_exposure import Exposure, load_by_file_prefix, load_by_filepath, scan_headers
from scipy.ndimage import gaussian_filter

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
    datestring = f[-1][0:4]+'-'+f[-1][4:6]+'-'+f[-1][6:8]
    filedatestring = f[-1][0:8]
    
    # Load first file that's not a guiding file (otherwise doesn't matter which it is) for device information
    non_guiding = [('guiding' not in f) for f in file_table['FILEPATH']]
    exp = Exposure(file_table[non_guiding][0]['FILEPATH'])
    cw = exp.col_width
    n_channels = exp.dev_size[0] // cw
    
    # Scan through files to find out what's available
    bias_present, dark_present, flat_present, longdark_present, singleframe_present = 0, 0, 0, 0, 0
    notes_present = 0
    for f in file_table['FILEPATH']:
        if 'bias' in f: bias_present = 1
        if 'guidingdark' in f: dark_present = 1
        if 'longdark' in f: longdark_present = 1
        if 'flat_' in f: flat_present = 1
        if 'singleframe' in f: singleframe_present = 1
    
    # Check for notes file
    if os.path.exists(dirname+'/analysis_notes.txt'):
        notes_present = 1
        notes_file = open(dirname+'/analysis_notes.txt')
        notes_lines = notes_file.readlines()
        notes_file.close()
        
    # Check for LED used from exposures taken under illumination
    is_ledwave = file_table['LEDWAVE'] != None
    if np.sum(is_ledwave) > 0:
        ledwave = np.unique(file_table['LEDWAVE'][is_ledwave].value)
    else:
        ledwave = None

    # Final summary numbers to generate
    read_noise, det_gain, dark_current, well_depth, read_time = {}, {}, {}, {}, {}
    read_noise_e, dark_current_e, well_depth_e = {}, {}, {}
    gain_modes = ['high (dual-gain)','low (dual-gain)','high','low']
    # A nominal gain value for converting ADU to electrons in case PTC gives bad values
    nom_gain = {'high (dual-gain)': 1.2, 'low (dual-gain)': 8.5, 'high': 1.2, 'low': 8.5}
    nom_read_noise = {'high (dual-gain)': 1.75, 'low (dual-gain)': 1.18, 'high': 1.75, 'low': 1.18}
    
    # Initialize bad pixel map and create summary filename based on whether or not we're using a subframe
    nowstring = time.strftime("%Y%m%d%H%M%S")
    if 'subframe' in kwargs:
        subframe = kwargs['subframe']
        x, y = subframe[1]-subframe[0]+1, subframe[3]-subframe[2]+1
        bad_pixel_map = np.zeros([x,y])
        doc_name = f'{filedatestring}_{camera}_{detid}{subframe}_analysis_report_{nowstring}.pdf'
    else:
        bad_pixel_map = np.zeros([exp.dev_size[1],exp.dev_size[0]])
        doc_name = f'{filedatestring}_{camera}_{detid}_analysis_report_{nowstring}.pdf'
        
    # Create a folder to hold output of analysis run inside the provided data directory
    output_dirname = os.path.join(dirname,f'{nowstring}_output')
    os.mkdir(output_dirname)
    
    ###################################################
    # Data analysis section
    # - Performs analysis tasks and writes result files
    ###################################################
    
    # Get frame readout time from the single-frame exposures
    if singleframe_present and notes_present:
        for i, l in enumerate(notes_lines):
            if l.startswith('Readout times:'):
                # A single frame readout lasts for a reset frame readout and an actual frame readout
                # So the actual time an exposure takes reading out is more-or-less half this amount
                read_time['high'] = float(notes_lines[i+1].split()[1]) / 2
                read_time['low'] = float(notes_lines[i+2].split()[1]) / 2
                read_time['hdr'] = float(notes_lines[i+3].split()[1]) / 2
    
    # Get bias frames, noise maps, and read noise measurements
    if bias_present:
        # Loop through the gain modes and load each bias frame individually using subframes
        # These files are *very* large
        med_bias_frames, noise_map, n_frame = {}, {}, {}
        for gain in ['hdr','high','low']:
            if 'subframe' in kwargs:
                # If a subframe is already defined, perform this on the subframe
                bifr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias_{gain}', **kwargs)[0]

                # Build the median bias and noise frames
                if gain == 'hdr':
                    med_bias_frames['high (dual-gain)'] = np.nanmedian(bifr.cds_frames[:,0],axis=0)
                    med_bias_frames['low (dual-gain)'] = np.nanmedian(bifr.cds_frames[:,1],axis=0)
                    noise_map['high (dual-gain)'] = np.nanstd(bifr.cds_frames[:,0],axis=0)
                    noise_map['low (dual-gain)'] = np.nanstd(bifr.cds_frames[:,1],axis=0)
                    n_frame['high (dual-gain)'], n_frame['low (dual-gain)'] = len(bifr.cds_frames), len(bifr.cds_frames)
                else:
                    med_bias_frames[gain] = np.nanmedian(bifr.cds_frames,axis=0)
                    noise_map[gain] = np.nanstd(bifr.cds_frames,axis=0)
                    n_frame[gain] = len(bifr.cds_frames)
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
                    
                    # Store number of frames used
                    n_frame['high (dual-gain)'] = len(bifr.cds_frames)
                    n_frame['low (dual-gain)'] = len(bifr.cds_frames)
                else:
                    med_bias_frames[gain] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    noise_map[gain] = np.zeros((exp.dev_size[1], exp.dev_size[0]))
                    for i in range(n_channels):
                        chsf = (i*cw,(i+1)*cw-1,0,exp.dev_size[1]-1) # Define subframe for this channel
                        bifr = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias_{gain}', subframe=chsf, **kwargs)[0]
                        
                        # Build the median bias and noise frames
                        med_bias_frames[gain][:,i*cw:(i+1)*cw] = np.nanmedian(bifr.cds_frames,axis=0)
                        noise_map[gain][:,i*cw:(i+1)*cw] = np.nanstd(bifr.cds_frames,axis=0)
                    
                    # Store number of frames used
                    n_frame[gain] = len(bifr.cds_frames)
                    
        # TODO: Store the median bias frames and add the frame stats to a table
        # The below needs as an input: a set of frames in dictionary form, indexed by gain
        # and a name for the file
        # That's literally it! Make a function.
        hdus = [fits.PrimaryHDU()]
        for g in med_bias_frames:
            imhdu = fits.ImageHDU(data=med_bias_frames[g], name=f'{g} bias frame')
            imhdu.header['GAIN'] = g
            hdus.append(imhdu)
        hdulist = fits.HDUList(hdus=hdus)
        hdulist.writeto(os.path.join(output_dirname,'bias_frames.fits'))
        # Output: med_bias_frame for each gain, noise_map for each gain, n_frame (# of frames) for each gain
        # Put them in output_dirname
        
    exit()
    
    
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
        if bias_present: summary_text += '- Bias frames\n'
        if longdark_present: summary_text += '- Long dark frames\n'
        if dark_present: summary_text += '- Standard operating dark frames\n'
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
        

        
        # Bias plots
        # Get bias frames
        # We expect one bias file with 100 frames per gain mode: high, low, HDR
        if bias_present:


            # Now create bias and noise map summary pages
            noise_map_e = {}
            for g in med_bias_frames:
                med_bias = med_bias_frames[g] * nom_gain[g]
                
                mmin, mmax = min(med_bias.flatten()), max(med_bias.flatten())
                
                summary_text = f'Gain mode: {g} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Median of {n_frame[g]} minimum-length exposures\n'
                summary_text += f'Min median pixel value: {mmin:.1f} e-; max median pixel value: {mmax:.1f} e- \n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, med_bias, f'Bias frames - gain: {g}', summary_text, precision=nom_gain[g])
                
                noise_map_e[g] = noise_map[g] * nom_gain[g]
                nmin, nmax = min(noise_map_e[g].flatten()), max(noise_map_e[g].flatten())
                
                # Define bad pixels as having noise above 3e- threshold in high gain
                bad_thresh = 3
                if (g == 'high') | (g == 'high (dual-gain)'):
                    percent_bad = np.sum(noise_map_e[g] >= bad_thresh) / noise_map_e[g].size * 100
                    bad_pixel_map[noise_map_e[g] >= bad_thresh] = 1
                
                # Calculate RMS read noise after filtering out catastrophically bad pixels
                read_noise[g] = np.sqrt(np.nanmean(noise_map[g][noise_map_e[g] < 100]**2))
                
                summary_text = f'Gain mode: {g} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Noise map over {n_frame[g]} minimum-length exposures\n'
                summary_text += f'Min noise value: {nmin:.1f} e-; max noise value: {nmax:.1f} e- \n'
                summary_text += f'Read noise (RMS): {read_noise[g]*nom_gain[g]:.2f} e-\n'
                if (g == 'high') | (g == 'high (dual-gain)'): summary_text += f'Percentage above {bad_thresh} e-: {percent_bad:.2f} %\n'
                
                # Also plot read noise map and histogram
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
                summary_text += f'RMS read noise, {g}: {read_noise[g]*nom_gain[g]:.2f} e-\n'
            plt.axvline(3,ls='--',color='grey',label='3e- Requirement')
            plt.xlabel('Read Noise (e-)')
            ax1.set_ylabel('Fraction of Pixels')
            min_y = 1 / (noise_map_e[g].size*2)
            ax1.set_ylim(min_y,1)
            ax2.set_ylabel('Cumulative Fraction of Pixels')
            ax2.set_ylim(0,1)
            ax1.loglog()
            plt.legend(fontsize=10,loc=1)
            
            # Also Fig 17 plot?
            
            plt.text(0.0, -0.3, summary_text, transform=ax1.transAxes, verticalalignment='top')
            
            plt.tight_layout()
            fig.text(0.96, 0.02, pdf.get_pagecount()+1)
            pdf.savefig()
            plt.close()
            
            bias_note = ', bias subtracted'
        else:
            bias_note = ''
        
        # Long darks
        if longdark_present:
            # Load long dark frames
            longdark_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_longdark', **kwargs)
            longdark = {'high (dual-gain)': [], 'low (dual-gain)': []}
            shortdark = {'high (dual-gain)': [], 'low (dual-gain)': []}
            long_exp_time, short_exp_time = 0, 0
            for longdark_frame in longdark_frames:
                if longdark_frame.exp_time > 1000:
                    # Collect the actual long dark frames
                    longdark['high (dual-gain)'].append(longdark_frame.cds_frames[0,0])
                    longdark['low (dual-gain)'].append(longdark_frame.cds_frames[0,1])
                    long_exp_time = longdark_frame.exp_time
                else:
                    # Collect any shorter frames for bias subtraction
                    shortdark['high (dual-gain)'].append(longdark_frame.cds_frames[0,0])
                    shortdark['low (dual-gain)'].append(longdark_frame.cds_frames[0,1])
                    short_exp_time = longdark_frame.exp_time
                
            for g in longdark:
                longdark_frames = np.array(longdark[g])
                shortdark_frames = np.array(shortdark[g])
                frame = np.median(longdark_frames,axis=0) * nom_gain[g] / long_exp_time
                
                # Subtract either a) a shorter dark frame or b) the bias frame or c) nothing (non-ideal scenario)
                if len(shortdark_frames) > 0:
                    exp_time = long_exp_time - short_exp_time
                    frame = (np.median(longdark_frames,axis=0) - np.median(shortdark_frames,axis=0)) * nom_gain[g] / exp_time
                    sub_note = f', {short_exp_time}s frame subtracted'
                elif bias_present:
                    exp_time = long_exp_time
                    frame = (np.median(longdark_frames,axis=0) - med_bias_frames[g]) * nom_gain[g] / exp_time
                    sub_note = f', bias subtracted'
                else:
                    exp_time = long_exp_time
                    frame = np.median(longdark_frames,axis=0) * nom_gain[g] / exp_time
                    sub_note = f', no bias subtracted'
                
                dmin, dmax, dmedian, dmean = min(frame.flatten()), max(frame.flatten()), np.median(frame), np.mean(frame)
                bad_thresh = 0.003
                percent_bad = np.sum(frame > bad_thresh) / frame.size * 100
                if (g == 'high') | (g == 'high (dual-gain)'): bad_pixel_map[frame > bad_thresh] = 2
                # Dark current 'measurement' is the 99% percentile
                dark_current[g] = np.percentile(frame,99)
                
                if 'subframe' not in kwargs:
                    # Boxcar smooth for plotting and histogram, if not looking at a subframe
                    smoothed_dark = convolve(frame, Box2DKernel(5))
                    smoothed_txt = ', boxcar-smoothed with width=5'
                    plot_dark = smoothed_dark
                else:
                    smoothed_txt = ''
                    plot_dark = frame
                
                summary_text = f'Frame: {g} long dark{sub_note} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Exposure time {exp_time} s{smoothed_txt}\n'
                summary_text += f'Min median pixel value: {dmin:.4f} e-/s; max median pixel value: {dmax:.4f} e-/s \n'
                summary_text += f'Median pixel value: {dmedian:.4f} e-/s; mean pixel value: {dmean:.4f} e-/s\n'
                if (g == 'high') | (g == 'high (dual-gain)'): summary_text += f'Percentage above {bad_thresh} e-/s: {percent_bad:.2f} %; 99% percentile: {dark_current[g]:.4f} e-/s\n'
                
                # Create a standard histogram page and save it to the pdf
                prec = nom_gain[g] * 0.0002
                if (g == 'high') | (g == 'high (dual-gain)'):
                    hist_page(pdf, plot_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec, contours=[bad_thresh], vlines=[bad_thresh])
                else:
                    hist_page(pdf, plot_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec)

        # Standard operating darks
        # Get darks
        if dark_present:
            for mode in ['FUVdark','NUVdark','NUVguidingdark']:
                if mode == 'FUVdark':
                    long = 900
                    short = 9
                else:
                    long = 300
                    short = 3
                    
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
                        long_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{long}_', **kwargs)[0]
                        short_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{short}_', **kwargs)[0]
                        
                        long_high.append(long_frame.cds_frames[0,0])
                        long_low.append(long_frame.cds_frames[0,1])
                        short_high.append(short_frame.cds_frames[0,0])
                        short_low.append(short_frame.cds_frames[0,1])
                        
                        l_starts.append(long_frame.date)
                        s_starts.append(short_frame.date)
                
                long_high = np.stack(long_high)
                long_low = np.stack(long_low)
                short_high = np.stack(short_high)
                short_low = np.stack(short_low)
                
                # For each resulting median frame, print a page of plots
                frames = [np.median(short_low,axis=0) * nom_gain['low (dual-gain)'] / short,
                          np.median(short_high,axis=0) * nom_gain['high (dual-gain)'] / short,
                          np.median(long_low,axis=0) * nom_gain['low (dual-gain)'] / long,
                          np.median(long_high,axis=0) * nom_gain['high (dual-gain)'] / long]
                names = ['short low-gain','short high-gain','long low-gain','long high-gain']
                precisions = [5, 0.72, 0.05, 0.0072]
                times = [short, short, long, long]
                
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
                sl_mean = np.mean(short_low, axis=(1,2)) * nom_gain['low (dual-gain)']
                sl_std = np.std(short_low, axis=(1,2)) * nom_gain['low (dual-gain)']
                sh_mean = np.mean(short_high, axis=(1,2)) * nom_gain['high (dual-gain)']
                sh_std = np.std(short_high, axis=(1,2)) * nom_gain['high (dual-gain)']
                ll_mean = np.mean(long_low, axis=(1,2)) * nom_gain['low (dual-gain)']
                ll_std = np.std(long_low, axis=(1,2)) * nom_gain['low (dual-gain)']
                lh_mean = np.mean(long_high, axis=(1,2)) * nom_gain['high (dual-gain)']
                lh_std = np.std(long_high, axis=(1,2)) * nom_gain['high (dual-gain)']
                
                if 'subframe' not in kwargs:
                    # Also show a column-by-column breakdown if not looking at a subframe
                    sl_col_mean, sh_col_mean, ll_col_mean, lh_col_mean = [], [], [], []
                    for i in range(n_channels):
                        sl_col_mean.append(np.mean(short_low[:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['low (dual-gain)'])
                        sh_col_mean.append(np.mean(short_high[:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['high (dual-gain)'])
                        ll_col_mean.append(np.mean(long_low[:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['low (dual-gain)'])
                        lh_col_mean.append(np.mean(long_high[:,:,i*cw:(i+1)*cw],axis=(1,2)) * nom_gain['high (dual-gain)'])
                    frame_label = 'Whole detector'
                else:
                    frame_label = f'Subframe: {kwargs["subframe"]}'
                
                # Convert observation times to seconds
                s_starts = [(s_starts[-1] - s).total_seconds() for s in s_starts]
                l_starts = [(l_starts[-1] - l).total_seconds() for l in l_starts]

                plt.sca(ax1)
                plt.title(f'{names[0]}')
                plt.plot(s_starts,sl_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts,sl_mean-sl_std,sl_mean+sl_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): p = plt.plot(s_starts,sl_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                plt.legend(fontsize=10,ncol=6,loc=9)
                
                plt.sca(ax2)
                plt.title(f'{names[1]}')
                plt.plot(s_starts,sh_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts,sh_mean-sh_std,sh_mean+sh_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(s_starts,sh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                # Mean long-frame dark value over time
                plt.sca(ax3)
                plt.title(f'{names[2]}')
                plt.plot(l_starts,ll_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts,ll_mean-ll_std,ll_mean+ll_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(l_starts,ll_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                plt.sca(ax4)
                plt.title(f'{names[3]}')
                plt.plot(l_starts,lh_mean,'x-',color='k',label=frame_label)
                plt.fill_between(s_starts,lh_mean-lh_std,lh_mean+lh_std,color='k',alpha=0.1, label='St. Dev.')
                if 'subframe' not in kwargs:
                    for j in range(n_channels): plt.plot(l_starts,lh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.xlabel('Time (s)')
                plt.ylabel('e-')
                
                plt.tight_layout()
                fig.text(0.96, 0.02, pdf.get_pagecount()+1)
                pdf.savefig()
                plt.close()
    
        if flat_present:
            mid_flats, mid_flat_times, mid_flat_voltages = {}, {}, {}
            exp_times, med_sig, var, voltage = {}, {}, {}, {}
            allflats = []
            for f in file_table['FILEPATH']:
                if 'flat' in f: allflats.append(True)
                else: allflats.append(False)
                
            for gain in ['hdr','high','low']:
                gain_flats = (file_table['GAIN'] == gain) & np.array(allflats)

                if np.sum(gain_flats) > 0:
                    # Load flat frame files
                    flat_files = file_table['FILEPATH'][gain_flats]
                    flat_frames = load_by_filepath(file_table['FILEPATH'][gain_flats], **kwargs)
                    exptime = file_table['EXPTIME'][gain_flats]
                    if gain in read_time: exptime = exptime + read_time[gain]
                    if (file_table['LED'][gain_flats] > -1).any():
                        led = file_table['LED'][gain_flats] # get LED voltage from FITS header
                    else:
                        led = [f.split('_')[4] for f in flat_files] # get LED voltage from filename
                    
                    led_vals = np.unique(led)
                    max_exp, max_i = max(exptime), np.argmax(exptime)
                    
                    # Check whether we want to do masking of existing bad pixels
                    # If >50% of pixels are bad, there's probably a deeper issue, so turn bad pixel masking off
                    bad_frac = np.sum((bad_pixel_map > 0)) / bad_pixel_map.size
                    if bad_frac < 0.5: domask = True
                    else: domask = False
                    
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

                    if gain == 'hdr':
                        medians, variance = np.zeros((1,2)), np.zeros((1,2))
                        all_exp_times, all_voltage = np.zeros(1), np.zeros(1)
                        
                        for i in range(x_parts):
                            for j in range(y_parts):
                                x1, x2, y1, y2 = x_size*i, x_size*(i+1), y_size*j, y_size*(j+1)
                                if domask:
                                    mask = bad_pixel_map[y1:y2,x1:x2]
                                    medians = np.append(medians, [ff.get_median((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                                    variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                                else:
                                    medians = np.append(medians, [ff.get_median((x1, x2, y1, y2)) for ff in flat_frames], axis=0)
                                    variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2)) for ff in flat_frames], axis=0)
                                all_exp_times = np.append(all_exp_times, exptime)
                                all_voltage = np.append(all_voltage, led)
                        medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                        exp_times['high (dual-gain)'], exp_times['low (dual-gain)'] = all_exp_times, all_exp_times
                        voltage['high (dual-gain)'], voltage['low (dual-gain)'] = all_voltage, all_voltage
                        med_sig['high (dual-gain)'], med_sig['low (dual-gain)'] = medians[:,0], medians[:,1]
                        var['high (dual-gain)'], var['low (dual-gain)'] = variance[:,0], variance[:,1]
                        
                        # Find the mid-range exposures - something close to 10000 ADU
                        whole_frame_medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        hmid_exp_i = np.argmin(np.abs(whole_frame_medians[:,0] - 10000))
                        lmid_exp_i = np.argmin(np.abs(whole_frame_medians[:,1] - 10000))

                        mid_flat_times['high (dual-gain)'], mid_flat_times['low (dual-gain)'] = exptime[hmid_exp_i], exptime[lmid_exp_i]
                        mid_flat_voltages['high (dual-gain)'], mid_flat_voltages['low (dual-gain)'] = led[hmid_exp_i], led[lmid_exp_i]
                        if bias_present:
                            mid_flats['high (dual-gain)'] = flat_frames[hmid_exp_i].cds_frames[0,0] - med_bias_frames['high (dual-gain)']
                            mid_flats['low (dual-gain)'] = flat_frames[lmid_exp_i].cds_frames[0,1] - med_bias_frames['low (dual-gain)']
                        else:
                            mid_flats['high (dual-gain)'] = flat_frames[hmid_exp_i].cds_frames[0,0]
                            mid_flats['low (dual-gain)'] = flat_frames[lmid_exp_i].cds_frames[0,1]

                    else:
                        # Loop through 256x256 chunks of the detector to get multiple subframe medians
                        medians, variance = np.zeros(1), np.zeros(1)
                        all_exp_times, all_voltage = np.zeros(1), np.zeros(1)
                        
                        for i in range(x_parts):
                            for j in range(y_parts):
                                x1, x2, y1, y2 = x_size*i, x_size*(i+1), y_size*j, y_size*(j+1)
                                if domask:
                                    mask = bad_pixel_map[y1:y2,x1:x2]
                                    medians = np.append(medians, [ff.get_median((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                                    variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2), mask=mask) for ff in flat_frames], axis=0)
                                else:
                                    medians = np.append(medians, [ff.get_median((x1, x2, y1, y2)) for ff in flat_frames], axis=0)
                                    variance = np.append(variance, [ff.get_variance((x1, x2, y1, y2)) for ff in flat_frames], axis=0)
                                all_exp_times = np.append(all_exp_times, exptime)
                                all_voltage = np.append(all_voltage, led)
                        medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                        exp_times[gain] = all_exp_times
                        voltage[gain] = all_voltage
                        med_sig[gain] = medians
                        var[gain] = variance
                        
                        # Find the mid-range exposures
                        whole_frame_medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        mid_exp_i = np.argmin(np.abs(whole_frame_medians - 10000))
                        mid_flat_times[gain] = exptime[mid_exp_i]
                        mid_flat_voltages[gain] = led[mid_exp_i]
                        if bias_present:
                            mid_flats[gain] = flat_frames[mid_exp_i].cds_frames[0] - med_bias_frames[gain]
                        else:
                            mid_flats[gain] = flat_frames[mid_exp_i].cds_frames[0]
            
            # Mid-range flats
            for g in mid_flats:
                flat = mid_flats[g] * nom_gain[g]
                mmin, mmax, mmedian, mmean = min(flat.flatten()), max(flat.flatten()), np.median(flat), np.mean(flat)
                # Define bad pixels as particularly low-gain, having 5-sigma lower signal than median pixel
                bad_thresh = mmean - 5*np.std(flat)
                percent_bad = np.sum(flat < bad_thresh) / flat.size * 100
                bad_pixel_map[flat < bad_thresh] = 3

                summary_text = f'Gain mode: {g}{bias_note}\n'
                summary_text += f'Illuminated {mid_flat_times[g]:.1f}s exposure; LED at {mid_flat_voltages[g]} V\n'
                summary_text += f'Min pixel value: {mmin:.1f} e-; max pixel value: {mmax:.1f} e- \n'
                summary_text += f'Median pixel value: {mmedian:.1f} e-; mean pixel value: {mmean:.1f} e-\n'
                summary_text += f'Percentage below {bad_thresh:.1f} e-/s (5-sigma below mean): {percent_bad:.2f} %\n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, flat, f'Mid-range flat frame - gain: {g}', summary_text, precision=min([50,mmax/50]))
                
            # Set up fitter and model (a PL set to linear solution)
            fit = fitting.LMLSQFitter(calc_uncertainties=True)
            fit_or = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=3.5)
            model = PowerLaw1D(amplitude=1, x_0=1, alpha=-1, fixed={'x_0': True, 'alpha': True})
            
            # Linearity and PTC plots
            
            # Linearity curve
            for g in med_sig:
                # Each gain gets its own page
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
                    for_fitting = (means > 10) & (means < (max(med_sig[g]) - 5000))
                    if np.sum(for_fitting) > 1:
                        model_fit, mask = fit_or(model, exposure_times[for_fitting], means[for_fitting], maxiter=100)
                        plt.plot(np.logspace(-0.3,2.4),model_fit(np.logspace(-0.3,2.4)), ls='--', color=gain_line_color[g])
                        if model_fit(0.8) > 0.5: plt.text(1.3,model_fit(0.8),f'{vol} V')
                        else: plt.text(150,model_fit(250),f'{vol} V')
                    
                        # Plot residuals from that fit altogether for each voltage as % from fit
                        plt.sca(vol_subplots[i])
                        plt.plot([0.5,30000],[0,0],ls='--',color='gray')
                        resid = (med_sig[g][this_voltage] - model_fit(exp_times[g][this_voltage])) / model_fit(exp_times[g][this_voltage]) * 100
                        resid_means = (means - model_fit(exposure_times)) / model_fit(exposure_times) * 100
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
                
                #plt.tight_layout()
                fig.text(0.96, 0.02, pdf.get_pagecount()+1)
                pdf.savefig()
                plt.close()
            
            # PTCs
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Photon Transfer Curve')
            
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))

            # PTC
            for g in med_sig:
                # Fit PTC for read noise level and gain
                if (g == 'high (dual-gain)') | (g == 'low (dual-gain)'):
                    plt.sca(ax1)
                else:
                    plt.sca(ax2)

                # Determine the shot noise dominated region of the PTC
                # Unsaturated and above the noise-floor-dominated region
                rough_sat_level = np.minimum(max(med_sig[g]) - 5000, 10000) # We expect saturation to be well above 10k but want to handle other cases
                valid = (var[g] > 0) & (med_sig[g] > 100) & (med_sig[g] < rough_sat_level)
                if np.sum(valid) > 0:
                    # Subtract the noise
                    if g in read_noise:
                        noise_floor = read_noise[g]**2
                    else:
                        noise_floor = nom_read_noise[g]**2
                    corrected_v = var[g] - noise_floor
                    for_fitting = valid & (corrected_v > 0)
                else:
                    # Nothing valid here for this gain, skip
                    continue
                
                try:
                    # Fit while rejecting outliers and not over-weighting high signal data points
                    model_fit, mask = fit_or(model, med_sig[g][for_fitting], corrected_v[for_fitting], maxiter=100)
                    
                    # Plot points used for fitting and points filtered out - mask filters out
                    plt.scatter(med_sig[g][for_fitting][mask], var[g][for_fitting][mask], marker='x', color=gain_color[g], alpha=alpha)
                    plt.scatter(med_sig[g][~for_fitting], var[g][~for_fitting], marker='x', color=gain_color[g], alpha=alpha)
                    plt.scatter(med_sig[g][for_fitting][~mask], var[g][for_fitting][~mask], marker='x', color=gain_color[g])

                    #K_ADC(e−/DN) is determined from the slope of the shot noise curve (i.e. powerlaw amplitude)
                    k = 1 / model_fit.amplitude.value
                    if fit_or.fit_info['param_cov']:
                        k_err = np.abs(k * -1 * np.sqrt(fit_or.fit_info['param_cov'][0][0]) / model_fit.amplitude.value)
                    else:
                        print('Error with calculating gain covariance')
                        k_err = 0
                    
                    # Plot fit
                    p = plt.plot(np.logspace(1,4.3), model_fit(np.logspace(1,4.3)), ls='--', color=gain_line_color[g],
                                 label=f'{g} gain: {k:.3f} ± {k_err:.3f} e-/ADU')
                    
                    # Plot noise and well depth
                    plt.plot([1,2e4],[noise_floor, noise_floor], ls='--', color=gain_line_color[g], alpha=0.5)
                    plt.plot([max(med_sig[g]),max(med_sig[g])],[1,max(var[g])], ls='--', color=gain_line_color[g], alpha=0.5)
                    
                    # Determine properties
                    det_gain[g] = [k, k_err]
                    if g in read_noise: read_noise[g] = read_noise[g] * k
                    if g in dark_current: dark_current[g] = dark_current[g] * k
                    well_depth[g] = np.nanmax(med_sig[g]) * k
                except:
                    print(f'Problem fitting slope for gain mode {g}, skipping and using nominal gain')
            
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
            pdf.savefig()
            plt.close()
        
        # So-what summary pages
        # Empty axes for final
        if bias_present and longdark_present and flat_present:
            # Bad pixel map page
            fig = plt.figure(figsize=[8.5,11],dpi=250)
            plt.suptitle('Bad pixel map')
            ax1 = plt.subplot2grid((5,1), (0,0), rowspan=2)
            plt.sca(ax1)
            bad_color_map = matplotlib.colors.ListedColormap(['white', 'red', 'blue', 'limegreen'])
            plt.imshow(bad_pixel_map+0.5, cmap=bad_color_map, vmin=0, vmax=4)
            percentage_bad = np.sum((bad_pixel_map > 0)) / bad_pixel_map.size * 100
            plt.text(0.0, -0.3, f'Number of bad pixels: {np.sum(bad_pixel_map)} ({percentage_bad:.2f} %)', transform=ax1.transAxes, verticalalignment='top')
            plt.colorbar(orientation='horizontal', label='Bad pixel type', ticks=[0.5,1.5,2.5,3.5],
                         format=mticker.FixedFormatter(['Good', 'High Noise', 'High Dark', 'Low Sensitivity']))
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
            
            for i, g in enumerate(gain_modes):
            
                # Determine detector properties where gain wasn't calculated
                if g not in det_gain:
                    if g in read_noise: read_noise[g] = read_noise[g] * nom_gain[g]
                    if g in dark_current: dark_current[g] = dark_current[g] * nom_gain[g]
            
                # Summary text
                plt.text(c1, 8.5-i*0.5, g)
                if g in det_gain: plt.text(c2, 8.5-i*0.5, f'{det_gain[g][0]:.3f}±{det_gain[g][1]:.3f}', ha='right')
                else: plt.text(c2, 8.5-i*0.5, f'{nom_gain[g]:.1f} (nom.)', ha='right')
                if g in read_noise: plt.text(c3, 8.5-i*0.5, f'{read_noise[g]:.2f}', ha='right')
                if g in dark_current: plt.text(c4, 8.5-i*0.5, f'{dark_current[g]*1000:.2f}', ha='right')
                if g in well_depth: plt.text(c5, 8.5-i*0.5, f'{int(well_depth[g])}', ha='right')
            
            fig.text(0.96, 0.02, pdf.get_pagecount()+1)
            pdf.savefig()
            plt.close()
            
        # TODO: something for guiding rows

def hist_page(pdf, data, title, summary_text, unit='e-', precision=1, contours=False, vlines=False):
    '''
    Default style of page showing a figure of the image, histograms, and summary text
    '''
    dmin, dmax = min(data.flatten()), max(data.flatten())
    data_range = int( (dmax - dmin) / precision )

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

