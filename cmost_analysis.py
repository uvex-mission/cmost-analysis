'''
    Functions relating to analysing the standard analysis exposures
    taken using the cmost_camera.py scripts. Operates on the same
    file naming scheme as cmost_camera.py so if that changes, this
    needs to change too
    
    Run this with Python 3
    
    Usage: python cmost_analysis.py data/20240413
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
    # Scan the FITS files in the data directory
    file_table = scan_headers(dirname,custom_keys=['LEDWAVE'])
    if not file_table:
        exit('No data files found in directory, closing')
    
    # Get camera and detector ID
    f = os.path.split(file_table[0]['FILEPATH'])[1].split('_')
    camera = f[0]
    detid = f[1]
    datestring = f[-1][0:4]+'-'+f[-1][4:6]+'-'+f[-1][6:8]
    
    # Load first file that's not a guiding file (otherwise doesn't matter which it is) for device information
    non_guiding = ['guiding' not in f for f in file_table['FILEPATH']]
    exp = Exposure(file_table[non_guiding][0]['FILEPATH'])
    
    # Scan through files to find out what's available
    bias_present, dark_present, flat_present, longdark_present, flatdark_present, singleframe_present = 0, 0, 0, 0, 0, 0
    notes_present = 0
    for f in file_table['FILEPATH']:
        if 'bias' in f: bias_present = 1
        if 'guidingdark' in f: dark_present = 1
        if 'longdark' in f: longdark_present = 1
        if 'flat_' in f: flat_present = 1
        if 'flatdark_' in f: flatdark_present = 1
        if 'singleframe' in f: singleframe_present = 1
    
    # Check for notes file
    if os.path.exists(dirname+'/analysis_notes.txt'):
        notes_present = 1
        notes_file = open(dirname+'/analysis_notes.txt')
        notes_lines = notes_file.readlines()
        notes_file.close()
        
    # Check for LED used
    is_ledwave = file_table['LEDWAVE'] != None
    if np.sum(is_ledwave) > 0:
        ledwave = np.unique(file_table['LEDWAVE'][is_ledwave].value)
    else:
        ledwave = None

    # Final summary numbers to generate
    read_noise, det_gain, dark_current, well_depth, read_time = {}, {}, {}, {}, {}
    gain_modes = ['high (dual-gain)','low (dual-gain)','high','low']
    # A nominal gain value for converting ADU to electrons (checked against PTC later)
    nom_gain = {'high (dual-gain)': 1.2, 'low (dual-gain)': 8.5, 'high': 1.2, 'low': 8.5}
    nom_read_noise = {'high (dual-gain)': 1.75, 'low (dual-gain)': 1.18, 'high': 1.75, 'low': 1.18}
    
    # Plot settings
    gain_color = {'high (dual-gain)': 'tab:blue', 'low (dual-gain)': 'tab:orange', 'high': 'tab:green', 'low': 'tab:red'}
    
    # Bad pixel map
    bad_pixel_map = np.zeros([exp.dev_size[1],exp.dev_size[0]])
    
    # Initialize report document
    with PdfPages(os.path.join(dirname,f'{camera}_{detid}_analysis_report_{time.strftime("%Y%m%d%H%M%S")}.pdf')) as pdf:
    #with PdfPages(os.path.join(dirname,f'analysis_report_test.pdf')) as pdf:
    
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    
        # Empty axes for first page
        fig = plt.figure(figsize=[8.5,11],dpi=250)
        plt.axis('off')
        
        # Temperature
        if exp.temperature > 0: temp = f'{exp.temperature:2f} K (measured)'
        else: temp = '140 K (assumed)'
        
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
        
        # Get frame readout time
        #single_present = 0
        if singleframe_present and notes_present:
            for i, l in enumerate(notes_lines):
                if l.startswith('Readout times:'):
                    # A single frame readout lasts for a 'ghost' frame readout and an actual frame readout
                    # So the actual time an exposure takes reading out is half this amount
                    read_time['high'] = float(notes_lines[i+1].split()[1]) / 2
                    read_time['low'] = float(notes_lines[i+2].split()[1]) / 2
                    hdr_time = float(notes_lines[i+3].split()[1]) / 2
                    read_time['high (dual-gain)'], read_time['low (dual-gain)'] = hdr_time, hdr_time
        
        # Bias plots
        # Get bias frames
        # We expect one bias file with 100 frames per gain mode: high, low, HDR
        #bias_present = 0
        if bias_present:
            bias_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias', **kwargs)
            
            med_bias_frames, var, noise_map = {}, {}, {}
            for bifr in bias_frames:
                # For each gain mode, print a page of plots
                if bifr.gain == 'hdr':
                    med_bias_frames['high (dual-gain)'] = np.median(bifr.cds_frames[:,0],axis=0)
                    med_bias_frames['low (dual-gain)'] = np.median(bifr.cds_frames[:,1],axis=0)
                    noise_map['high (dual-gain)'] = np.std(bifr.cds_frames[:,0],axis=0)
                    noise_map['low (dual-gain)'] = np.std(bifr.cds_frames[:,1],axis=0)
                else:
                    med_bias_frames[bifr.gain] = np.median(bifr.cds_frames,axis=0)
                    noise_map[bifr.gain] = np.std(bifr.cds_frames,axis=0)
                n_frame = len(bifr.cds_frames)

            noise_map_e = {}
            for g in med_bias_frames:
                med_bias = med_bias_frames[g] * nom_gain[g]
                
                mmin, mmax = min(med_bias.flatten()), max(med_bias.flatten())
                
                summary_text = f'Gain mode: {g} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Median of {n_frame} minimum-length exposures\n'
                summary_text += f'Min median pixel value: {mmin:.1f} e-; max median pixel value: {mmax:.1f} e- \n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, med_bias, f'Bias frames - gain: {g}', summary_text, precision=nom_gain[g])
                
                noise_map_e[g] = noise_map[g] * nom_gain[g]
                nmin, nmax = min(noise_map_e[g].flatten()), max(noise_map_e[g].flatten())
                
                # Define bad pixels as having catastrophically bad noise (say 100 e-)
                bad_thresh = 100
                percent_bad = np.sum(noise_map_e[g] >= bad_thresh) / noise_map_e[g].size * 100
                bad_pixel_map[noise_map_e[g] >= bad_thresh] = 1
                
                # Calculate RMS read noise after filtering out catastrophically bad pixels
                read_noise[g] = np.sqrt(np.mean(noise_map[g][noise_map_e[g] < bad_thresh]**2))
                
                summary_text = f'Gain mode: {g} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Min noise value: {nmin:.1f} e-; max noise value: {nmax:.1f} e- \n'
                summary_text += f'Read noise (RMS): {read_noise[g]*nom_gain[g]:.2f} e-\n'
                summary_text += f'Percentage above {bad_thresh} e-: {percent_bad:.2f} %\n'
                
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
            ax1.set_ylim(5e-8,1)
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
        #longdark_present = 0
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
                if (g == 'high') | (g == 'high (dual-gain)'): bad_pixel_map[frame > bad_thresh] = 1
                # Dark current 'measurement' is the 99% percentile
                dark_current[g] = np.percentile(frame,99)
                
                summary_text = f'Frame: {g} long dark{sub_note} (nominal gain {nom_gain[g]} e-/ADU)\n'
                summary_text += f'Exposure time {exp_time} s, boxcar-smoothed with width=5\n'
                summary_text += f'Min median pixel value: {dmin:.4f} e-/s; max median pixel value: {dmax:.4f} e-/s \n'
                summary_text += f'Median pixel value: {dmedian:.4f} e-/s; mean pixel value: {dmean:.4f} e-/s\n'
                summary_text += f'Percentage above {bad_thresh} e-/s: {percent_bad:.2f} %; 99% percentile: {dark_current[g]:.4f} e-/s\n'
                
                # Boxcar smooth for plotting and histogram
                smoothed_dark = convolve(frame, Box2DKernel(5))
                
                # Create a standard histogram page and save it to the pdf
                prec = nom_gain[g] * 0.0002
                if (g == 'high') | (g == 'high (dual-gain)'):
                    hist_page(pdf, smoothed_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec, contours=[bad_thresh], vlines=[bad_thresh])
                else:
                    hist_page(pdf, smoothed_dark, f'Long darks - {g} frame', summary_text, unit='e-/s', precision=prec)

        # Standard operating darks
        # Get darks
        #dark_present = 0
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
                    hist_page(pdf, med_dark, f'Dark frames - {mode} mode - {names[j]} frame', summary_text, precision=precisions[j])
                
                # Also make a page of mean values over time
                fig = plt.figure(figsize=[8.5,11],dpi=250)
                plt.suptitle(f'Dark frames - {mode} mode - mean over time')
                
                ax1 = plt.subplot2grid((4,1), (0,0))
                ax2 = plt.subplot2grid((4,1), (1,0))
                ax3 = plt.subplot2grid((4,1), (2,0))
                ax4 = plt.subplot2grid((4,1), (3,0))
                
                # Mean short-frame dark value over time (whole frame + every column)
                sl_col_mean, sh_col_mean,ll_col_mean, lh_col_mean = [], [], [], []
                n_cols = exp.dev_size[0]//256
                for i in range(n_cols):
                    sl_col_mean.append(np.mean(short_low[:,:,i*256:(i+1)*256],axis=(1,2)) * nom_gain['low (dual-gain)'])
                    sh_col_mean.append(np.mean(short_high[:,:,i*256:(i+1)*256],axis=(1,2)) * nom_gain['high (dual-gain)'])
                    ll_col_mean.append(np.mean(long_low[:,:,i*256:(i+1)*256],axis=(1,2)) * nom_gain['low (dual-gain)'])
                    lh_col_mean.append(np.mean(long_high[:,:,i*256:(i+1)*256],axis=(1,2)) * nom_gain['high (dual-gain)'])
                
                # Convert observation times to seconds
                s_starts = [(s_starts[-1] - s).total_seconds() for s in s_starts]
                l_starts = [(l_starts[-1] - l).total_seconds() for l in l_starts]

                plt.sca(ax1)
                plt.title(f'{names[0]}')
                plt.plot(s_starts,np.mean(short_low,axis=(1,2)) * nom_gain['low (dual-gain)'],'x-',color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(s_starts,sl_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                plt.legend(fontsize=10,ncol=6,loc=9)
                
                plt.sca(ax2)
                plt.title(f'{names[1]}')
                plt.plot(s_starts,np.mean(short_high,axis=(1,2)) * nom_gain['high (dual-gain)'],'x-',color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(s_starts,sh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                # Mean long-frame dark value over time
                plt.sca(ax3)
                plt.title(f'{names[2]}')
                plt.plot(l_starts,np.mean(long_low,axis=(1,2)) * nom_gain['low (dual-gain)'],'x-',color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(l_starts,ll_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.ylabel('e-')
                
                plt.sca(ax4)
                plt.title(f'{names[3]}')
                plt.plot(l_starts,np.mean(long_high,axis=(1,2)) * nom_gain['high (dual-gain)'],'x-',color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(l_starts,lh_col_mean[j],'x-',alpha=0.5,label=f'Col {j}')
                plt.xlabel('Time (s)')
                plt.ylabel('e-')
                
                plt.tight_layout()
                fig.text(0.96, 0.02, pdf.get_pagecount()+1)
                pdf.savefig()
                plt.close()
    
        #flat_present = 0
        if flat_present:
            sat_flats, mid_flats, mid_flat_times, mid_flat_voltages = {}, {}, {}, {}
            exp_times, med_sig, var, voltage = {}, {}, {}, {}
            for gain in ['hdr','high','low']:
                gain_flats = (file_table['GAIN'] == gain) & (file_table['LED'] > 0)
                
                if np.sum(gain_flats) > 0:
                    flat_files = file_table['FILEPATH'][gain_flats]
                    flat_frames = load_by_filepath(file_table['FILEPATH'][gain_flats], **kwargs)
                    exptime = file_table['EXPTIME'][gain_flats]
                    led = file_table['LED'][gain_flats]
                    
                    led_vals = np.unique(led)
                    max_exp, max_i = max(exptime), np.argmax(exptime)

                    if gain == 'hdr':
                        # Loop through 256x256 chunks of the detector to get multiple subframe medians
                        medians, variance = np.zeros((1,2)), np.zeros((1,2))
                        all_exp_times, all_voltage = np.zeros(1), np.zeros(1)
                        n_subframes = (exp.dev_size[0] // 256) * (exp.dev_size[1] // 256)
                        for i in range(exp.dev_size[0] // 256):
                            for j in range(exp.dev_size[1] // 256):
                                medians = np.append(medians, [ff.get_median((256*i,256*(i+1),256*j,256*(j+1))) for ff in flat_frames], axis=0)
                                variance = np.append(variance, [ff.get_variance((256*i,256*(i+1),256*j,256*(j+1))) for ff in flat_frames], axis=0)
                                all_exp_times = np.append(all_exp_times, exptime)
                                all_voltage = np.append(all_voltage, led)
                        medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                        '''
                        medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        variance = np.array([ff.get_variance((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        '''

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
                        n_subframes = (exp.dev_size[0] // 256) * (exp.dev_size[1] // 256)
                        for i in range(exp.dev_size[0] // 256):
                            for j in range(exp.dev_size[1] // 256):
                                medians = np.append(medians, [ff.get_median((256*i,256*(i+1),256*j,256*(j+1))) for ff in flat_frames], axis=0)
                                variance = np.append(variance, [ff.get_variance((256*i,256*(i+1),256*j,256*(j+1))) for ff in flat_frames], axis=0)
                                all_exp_times = np.append(all_exp_times, exptime)
                                all_voltage = np.append(all_voltage, led)
                        medians, variance, all_exp_times, all_voltage = medians[1:], variance[1:], all_exp_times[1:], all_voltage[1:]
                        '''
                        medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        variance = np.array([ff.get_variance((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                        '''
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
                bad_pixel_map[flat < bad_thresh] = 1

                summary_text = f'Gain mode: {g}{bias_note}\n'
                summary_text += f'Illuminated {mid_flat_times[g]}s exposure; LED at {mid_flat_voltages[g]} V\n'
                summary_text += f'Min pixel value: {mmin:.1f} e-; max pixel value: {mmax:.1f} e- \n'
                summary_text += f'Median pixel value: {mmedian:.1f} e-; mean pixel value: {mmean:.1f} e-\n'
                summary_text += f'Percentage below {bad_thresh:.1f} e-/s (5-sigma below mean): {percent_bad:.2f} %\n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, flat, f'Mid-range flat frame - gain: {g}', summary_text, precision=min([50,mmax/50]))
            
            # Linearity and PTC plots
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Linearity and photon transfer curves')
            
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
            
            # Linearity curve
            plt.sca(ax1)
            plt.title(f'Linearity')
            for g in med_sig:
                for vol in led_vals:
                    this_voltage_valid = (voltage[g] == vol) & (exp_times[g] > 0) & (med_sig[g] > 0) #& (var[g] > 0)
                    sort_i = np.argsort(exp_times[g][this_voltage_valid])
                    # Correct exposure times for the readout time
                    if g in read_time: exptime = exp_times[g][this_voltage_valid][sort_i] + read_time[g]
                    else: exptime = exp_times[g][this_voltage_valid][sort_i]
                    plt.plot(exptime, med_sig[g][this_voltage_valid][sort_i], 'x--', label=f'{g}', color=gain_color[g])
            plt.xlabel('Exposure Time (s)')
            plt.ylabel('Median Signal (ADU)')
            plt.loglog()
            # Avoid duplicate legend labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=10,ncol=4,loc=8,bbox_to_anchor=(0.5, -0.2))

            # PTC
            fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
            fit_or = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=3.5)
            plt.sca(ax2)
            plt.title(f'Photon Transfer Curve')
            for g in med_sig:
                # Fit PTC for read noise level and gain

                # Determine the shot noise dominated region of the PTC
                # Unsaturated and above the noise-floor-dominated region
                valid = (var[g] > 0) & (med_sig[g] > 100) & (med_sig[g] < 10000)
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

                model = PowerLaw1D(amplitude=1, x_0=1, alpha=-1, fixed={'x_0': True, 'alpha': True})
                
                try:
                    # Fit while rejecting outliers and not over-weighting high signal data points
                    model_fit, mask = fit_or(model, med_sig[g][for_fitting], corrected_v[for_fitting], maxiter=100)
                    
                    # Plot points used for fitting and points filtered out - mask filters out
                    plt.scatter(med_sig[g][for_fitting][mask], var[g][for_fitting][mask], marker='x', color=gain_color[g], alpha=0.05)
                    plt.scatter(med_sig[g][~for_fitting], var[g][~for_fitting], marker='x', color=gain_color[g], alpha=0.05)
                    plt.scatter(med_sig[g][for_fitting][~mask], var[g][for_fitting][~mask], marker='x', color=gain_color[g])

                    #K_ADC(e−/DN) is determined from the slope of the shot noise curve (i.e. powerlaw amplitude)
                    k = 1 / model_fit.amplitude.value
                    if fit_or.fit_info['param_cov']:
                        k_err = np.abs(k * -1 * np.sqrt(fit_or.fit_info['param_cov'][0][0]) / model_fit.amplitude.value)
                    else:
                        print('Error with calculating gain covariance')
                        k_err = 0
                        
                    # Plot fit
                    p = plt.plot(np.logspace(1,4.3), model_fit(np.logspace(1,4.3)), ls='--', label=f'{g} gain: {k:.3f} ± {k_err:.3f} e-/ADU')
                    
                    # Plot noise and well depth
                    plt.plot([1,3e4],[noise_floor, noise_floor], ls='--', color='grey')
                    plt.plot([max(med_sig[g]),max(med_sig[g])],[1,5e6], ls='--', color='grey')
                    
                    # Determine properties
                    det_gain[g] = [k, k_err]
                    if g in read_noise: read_noise[g] = read_noise[g] * k
                    if g in dark_current: dark_current[g] = dark_current[g] * k
                    well_depth[g] = max(med_sig[g]) * k
                except:
                    print(f'Problem fitting slope for gain mode {g}, skipping and using nominal gain')
                    
            plt.legend(loc=2)
            plt.xlabel('Median Signal (ADU)')
            plt.ylabel('Variance (ADU^2)')
            plt.loglog()

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
            plt.imshow(bad_pixel_map, cmap='Reds')
            percentage_bad = np.sum(bad_pixel_map) / bad_pixel_map.size * 100
            plt.text(0.0, -0.3, f'Number of bad pixels: {np.sum(bad_pixel_map)} ({percentage_bad:.2f} %)', transform=ax1.transAxes, verticalalignment='top')
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
    plt.imshow(data, vmin=np.percentile(data,0.01), vmax=np.percentile(data,99.9))
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
    python [-g] cmost_analysis.py DIRECTORY
    '''
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    
    if len(args) < 1:
        dirname = input('Analysis directory: ')
    else:
        dirname = args[0]
    
    kwargs = {}
    if '-g' in opts:
        kwargs['graycode'] = True

    standard_analysis_products(dirname,**kwargs)

