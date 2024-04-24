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
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmost_exposure import Exposure, load_by_file_prefix

font = {'size' : 12, 'family' : 'sans-serif'}
matplotlib.rc('font', **font)

def standard_analysis_products(dirname):
    '''
    Load standard exposures from the given directory and output
    standard analysis reports
    '''
    # Load file list
    files = os.listdir(dirname)
    if len(files) < 1:
        print('No data files found in directory')
        return False
    
    # Get camera and detector ID
    f = files[0].split('_')
    camera = f[0]
    detid = f[1]
    datestring = f[-1][0:4]+'-'+f[-1][4:6]+'-'+f[-1][6:8]
    
    # Load first file (doesn't matter which it is) for device information
    exp = Exposure(os.path.join(dirname,files[0]))
    
    # Scan through files to find out what's available
    for f in files:
        if 'bias' in f: bias_present = 1
        if 'dark' in f: dark_present = 1
        if 'flat' in f: flat_present = 1
    
    # Initialize report document
    with PdfPages(os.path.join(dirname,f'{camera}_{detid}_analysis_report.pdf')) as pdf:
    
        # Empty axes for first page
        fig = plt.figure(figsize=[8.5,11],dpi=300)
        plt.axis('off')
        
        # Device and test summary text
        summary_text = f'Cosmetic report for DeviceID: {detid}\n'
        summary_text += f'Device size: {exp.dev_size[0]} x {exp.dev_size[1]}\n\n'
        summary_text += f'Standard analysis exposures taken {datestring} using {camera} camera\n\n'
        summary_text += 'Contents:\n'
        if bias_present: summary_text += '- Bias frames\n'
        if dark_present: summary_text += '- Standard operating dark frames\n'
        if flat_present: summary_text += '- Flat fields vs exposure time\n'
        
        plt.text(0,1,summary_text,verticalalignment='top')
        pdf.savefig()
        plt.close()
        
        # Bias plots
        # Get bias frames
        # We expect one bias frame per gain mode: high, low, HDR
        #bias_present = 0
        if bias_present:
            bias_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias')
            for bifr in bias_frames:
                # For each gain mode, print a page of plots
                if bifr.gain == 'hdr':
                    gain = ['high (dual-gain)', 'low (dual-gain)']
                    med_frames = [np.median(bifr.cds_frames[:,0],axis=0), np.median(bifr.cds_frames[:,1],axis=0)]
                else:
                    gain = [bifr.gain]
                    med_frames = [np.median(bifr.cds_frames,axis=0)]
                n_frame = len(bifr.cds_frames)
                
                for i, med_bias in enumerate(med_frames):
                    mmin, mmax = min(med_bias.flatten()), max(med_bias.flatten())
                    med_range = int(np.ceil(mmax) - np.floor(mmin))

                    fig = plt.figure(figsize=[8.5,11],dpi=300)
                    plt.suptitle(f'Bias frames - gain: {gain[i]}')
                
                    bias_hist, bin_edges = np.histogram(med_bias,bins=100)
                    bias_hist2, bin_edges2 = np.histogram(med_bias,bins=med_range) # i.e. binsize=1

                    # Output a median frame image and histogram
                    ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2, colspan=2)
                    ax2 = plt.subplot2grid((5,2), (2,0), rowspan=2)
                    ax3 = plt.subplot2grid((5,2), (2,1), rowspan=2)
                    
                    plt.sca(ax1)
                    plt.imshow(med_bias, vmin=0, vmax=50)
                    plt.colorbar(label='ADU', shrink=0.9)
                    
                    plt.sca(ax2)
                    plt.stairs(bias_hist,edges=bin_edges)
                    plt.xlabel('ADU')
                    plt.ylabel('Pixel Count')
                    plt.text(0.5,0.9,'Whole range',transform=ax2.transAxes)
                    plt.semilogy()
                    
                    plt.sca(ax3)
                    plt.stairs(bias_hist2,edges=bin_edges2)
                    plt.xlabel('ADU')
                    plt.xlim(np.percentile(med_bias,[1.e-2,100-1.e-2]))
                    plt.text(0.3,0.9,'0.01 - 99.99 percentile',transform=ax3.transAxes)
                    plt.semilogy()
                    
                    summary_text = f'Gain mode: {gain[i]}\n'
                    summary_text += f'Median of {n_frame} minimum-length exposures\n'
                    summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU'
                    
                    plt.text(0.0, -0.3, summary_text, transform=ax2.transAxes, verticalalignment='top')
                
                    plt.subplots_adjust(hspace=0.25)
                    pdf.savefig()
                    plt.close()
            
        # Standard operating darks
        # Get darks
        if dark_present:
            for mode in ['FUVdark','NUVdark','NUVdarkguiding']:
                if mode == 'FUVdark':
                    long = 900
                    short = 9
                else:
                    long = 300
                    short = 3
                    
                # Load the ten frames in temporal order
                short_low, long_low, long_high = [], [], []
                long_guideframes, short_guideframes = [], []
                s_starts, l_starts = [], []
                n_frame = 10
                for i in range(n_frame):
                    if mode == 'NUVdarkguiding':
                        long_data = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{long}_')
                        short_data = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{short}_')
                        for l in long_data:
                            if 'guideframes' in l.filepath:
                                long_guideframes.append(l.cds_frames)
                            else:
                                long_frame = l
                        for s in short_data:
                            if 'guideframes' in s.filepath:
                                short_guideframes.append(s.cds_frames)
                            else:
                                short_frame = s
                    else:
                        long_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{long}_')[0]
                        short_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{short}_')[0]
                        
                    long_high.append(long_frame.cds_frames[0,0])
                    long_low.append(long_frame.cds_frames[0,1])
                    l_starts.append(long_frame.date)
                        
                    short_low.append(short_frame.cds_frames[0])
                    s_starts.append(short_frame.date)
                
                long_high = np.stack(long_high)
                long_low = np.stack(long_low)
                short_low = np.stack(short_low)
                
                # For each resulting median frame, print a page of plots
                frames = [np.median(short_low,axis=0), np.median(long_low,axis=0), np.median(long_high,axis=0)]
                names = ['short low-gain','long low-gain','long high-gain']
                times = [short, long, long]
                
                for j, med_dark in enumerate(frames):
                    mmin, mmax = min(med_dark.flatten()), max(med_dark.flatten())
                    med_range = int(np.ceil(mmax) - np.floor(mmin))

                    fig = plt.figure(figsize=[8.5,11],dpi=300)
                    plt.suptitle(f'Dark frames - {mode} mode - {names[j]} frame')
                
                    dark_hist, bin_edges = np.histogram(med_dark,bins=100)
                    dark_hist2, bin_edges2 = np.histogram(med_dark,bins=med_range) # i.e. binsize=1

                    # Output a median frame image and histogram
                    ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2, colspan=2)
                    ax2 = plt.subplot2grid((5,2), (2,0), rowspan=2)
                    ax3 = plt.subplot2grid((5,2), (2,1), rowspan=2)
                    
                    plt.sca(ax1)
                    plt.imshow(med_dark, vmin=0, vmax=50)
                    plt.colorbar(label='ADU', shrink=0.9)
                    
                    plt.sca(ax2)
                    plt.stairs(dark_hist,edges=bin_edges)
                    plt.xlabel('ADU')
                    plt.ylabel('Pixel Count')
                    plt.text(0.5,0.9,'Whole range',transform=ax2.transAxes)
                    plt.semilogy()
                    
                    plt.sca(ax3)
                    plt.stairs(dark_hist2,edges=bin_edges2)
                    plt.xlabel('ADU')
                    plt.xlim(np.percentile(med_dark,[1.e-2,100-1.e-2]))
                    plt.text(0.3,0.9,'0.01 - 99.99 percentile',transform=ax3.transAxes)
                    plt.semilogy()
                    
                    summary_text = f'Frame: {names[j]}\n'
                    summary_text += f'Median of {n_frame} x {times[j]}s exposures\n'
                    summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU'
                    
                    plt.text(0.0, -0.3, summary_text, transform=ax2.transAxes, verticalalignment='top')
                
                    plt.subplots_adjust(hspace=0.25)
                    pdf.savefig()
                    plt.close()
                
                
                # Also make a page of mean values over time
                fig = plt.figure(figsize=[8.5,11],dpi=300)
                plt.suptitle(f'Dark frames - {mode} mode - mean over time')
                
                ax1 = plt.subplot2grid((3,1), (0,0))
                ax2 = plt.subplot2grid((3,1), (1,0))
                ax3 = plt.subplot2grid((3,1), (2,0))
                
                # Mean short-frame dark value over time (whole frame + every column)
                sl_col_mean, ll_col_mean, lh_col_mean = [], [], []
                n_cols = exp.dev_size[0]//256
                for i in range(n_cols):
                    sl_col_mean.append(np.mean(short_low[:,:,i*256:(i+1)*256],axis=(1,2)))
                    ll_col_mean.append(np.mean(long_low[:,:,i*256:(i+1)*256],axis=(1,2)))
                    lh_col_mean.append(np.mean(long_high[:,:,i*256:(i+1)*256],axis=(1,2)))

                plt.sca(ax1)
                plt.title(f'{names[0]}')
                #plt.errorbar(s_starts,np.mean(short_low,axis=(1,2)),np.std(short_low,axis=(1,2)))
                plt.plot(s_starts,np.mean(short_low,axis=(1,2)),color='k')
                for j in range(n_cols): plt.plot(s_starts,sl_col_mean[j],alpha=0.5)
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # Mean long-frame dark value over time
                plt.sca(ax2)
                plt.title(f'{names[1]}')
                plt.plot(l_starts,np.mean(long_low,axis=(1,2)),color='k')
                for j in range(n_cols): plt.plot(l_starts,ll_col_mean[j],alpha=0.5)
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                plt.sca(ax3)
                plt.title(f'{names[2]}')
                plt.plot(l_starts,np.mean(long_high,axis=(1,2)),color='k')
                for j in range(n_cols): plt.plot(l_starts,lh_col_mean[j],alpha=0.5)
                plt.xlabel('Time')
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
    
    # TODO: something for guiding rows
    # TODO: optimize color bars
    # TODO: flat report pages



if __name__ == '__main__':
    '''
    Usage:
    python cmost_analysis.py DIRECTORY
    '''
    if len(sys.argv) < 2:
        dirname = input('Analysis directory: ')
    else:
        dirname = sys.argv[1]
        
    standard_analysis_products(dirname)

