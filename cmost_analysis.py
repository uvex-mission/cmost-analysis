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

def standard_analysis_products(dirname, **kwargs):
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
    bias_present, dark_present, flat_present = 0, 0, 0
    for f in files:
        if 'bias' in f: bias_present = 1
        if 'dark' in f: dark_present = 1
        if 'flat' in f: flat_present = 1
    
    # Initialize report document
    with PdfPages(os.path.join(dirname,f'{camera}_{detid}_analysis_report.pdf')) as pdf:
    
        # Empty axes for first page
        fig = plt.figure(figsize=[8.5,11],dpi=250)
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
            bias_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_bias', **kwargs)
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
                    n_bad = np.sum(med_bias > 50)
                    bad_percent = n_bad / med_bias.size * 100.
                    
                    summary_text = f'Gain mode: {gain[i]}\n'
                    summary_text += f'Median of {n_frame} minimum-length exposures\n'
                    summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU\n'
                    summary_text += f'Pixels > 50 ADU: {n_bad}, {bad_percent:.3f}%\n'
                    
                    # Create a standard histogram page and save it to the pdf
                    hist_page(pdf, med_bias, f'Bias frames - gain: {gain[i]}', summary_text)
            
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
                            
                            long_high.append(long_frame.cds_frames[0,0])
                            long_low.append(long_frame.cds_frames[0,1])
                            l_starts.append(long_frame.date)
                            
                            short_high.append(short_frame.cds_frames[0,0])
                            short_low.append(short_frame.cds_frames[0,1])
                            s_starts.append(short_frame.date)
                            
                            # TODO: also load guiding frames
                else:
                    n_frame = 9
                    for i in range(n_frame):
                        long_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{long}_', **kwargs)[0]
                        short_frame = load_by_file_prefix(f'{dirname}/{camera}_{detid}_{mode}{i}_{short}_', **kwargs)[0]
                            
                        long_high.append(long_frame.cds_frames[0,0])
                        long_low.append(long_frame.cds_frames[0,1])
                        l_starts.append(long_frame.date)
                            
                        short_high.append(short_frame.cds_frames[0,0])
                        short_low.append(short_frame.cds_frames[0,1])
                        s_starts.append(short_frame.date)
                
                long_high = np.stack(long_high)
                long_low = np.stack(long_low)
                short_high = np.stack(short_high)
                short_low = np.stack(short_low)
                
                # For each resulting median frame, print a page of plots
                frames = [np.median(short_low,axis=0), np.median(short_high,axis=0), np.median(long_low,axis=0), np.median(long_high,axis=0)]
                names = ['short low-gain','short high-gain','long low-gain','long high-gain']
                times = [short, short, long, long]
                
                for j, med_dark in enumerate(frames):
                    mmin, mmax = min(med_dark.flatten()), max(med_dark.flatten())
                    
                    summary_text = f'Frame: {names[j]}\n'
                    summary_text += f'Median of {n_frame} x {times[j]}s exposures\n'
                    summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU'
                    
                    # Create a standard histogram page and save it to the pdf
                    hist_page(pdf, med_dark, f'Dark frames - {mode} mode - {names[j]} frame', summary_text)
                
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
                    sl_col_mean.append(np.mean(short_low[:,:,i*256:(i+1)*256],axis=(1,2)))
                    sh_col_mean.append(np.mean(short_high[:,:,i*256:(i+1)*256],axis=(1,2)))
                    ll_col_mean.append(np.mean(long_low[:,:,i*256:(i+1)*256],axis=(1,2)))
                    lh_col_mean.append(np.mean(long_high[:,:,i*256:(i+1)*256],axis=(1,2)))

                plt.sca(ax1)
                plt.title(f'{names[0]}')
                #plt.errorbar(s_starts,np.mean(short_low,axis=(1,2)),np.std(short_low,axis=(1,2)))
                plt.plot(s_starts,np.mean(short_low,axis=(1,2)),color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(s_starts,sl_col_mean[j],alpha=0.5,label=f'Col {j}')
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.legend(fontsize=10,ncol=6,loc=9)
                #plt.legend(fontsize=10,ncol=6,loc=8,bbox_to_anchor=(0.5, -0.25))
                
                plt.sca(ax2)
                plt.title(f'{names[1]}')
                plt.plot(s_starts,np.mean(short_high,axis=(1,2)),color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(s_starts,sh_col_mean[j],alpha=0.5,label=f'Col {j}')
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # Mean long-frame dark value over time
                plt.sca(ax3)
                plt.title(f'{names[2]}')
                plt.plot(l_starts,np.mean(long_low,axis=(1,2)),color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(l_starts,ll_col_mean[j],alpha=0.5,label=f'Col {j}')
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                plt.sca(ax4)
                plt.title(f'{names[3]}')
                plt.plot(l_starts,np.mean(long_high,axis=(1,2)),color='k',label='Whole chip')
                for j in range(n_cols): plt.plot(l_starts,lh_col_mean[j],alpha=0.5,label=f'Col {j}')
                plt.xlabel('Time')
                plt.ylabel('ADU')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
    
        if flat_present:
            gain_label, sat_flats, mid_flats, mid_flat_times = [], [], [], []
            exp_times, med_sig, var = [], [], []
            for gain in ['hdr','high','low']:
                flat_frames = load_by_file_prefix(f'{dirname}/{camera}_{detid}_flat_{gain}', **kwargs)
                exp = np.array([ff.exp_time for ff in flat_frames])
                max_exp, max_i = max(exp), np.argmax(exp)

                if gain == 'hdr':
                    gain_label.extend(['high (dual-gain)', 'low (dual-gain)'])
                    sat_flats.extend((flat_frames[max_i].cds_frames[0,0], flat_frames[max_i].cds_frames[0,1]))
                    
                    medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                    variance = np.array([ff.get_variance((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                    
                    exp_times.extend((exp,exp))
                    med_sig.extend((medians[:,0],medians[:,1]))
                    var.extend((variance[:,0],variance[:,1]))
                    
                    # Find the mid-range exposures
                    hmid_exp_i = np.where( (medians[:,0] > 5000) & (medians[:,0] < 15000) )[0][0]
                    lmid_exp_i = np.where( (medians[:,1] > 5000) & (medians[:,1] < 15000) )[0][0]
                    
                    mid_flat_times.extend([exp[hmid_exp_i], exp[lmid_exp_i]])
                    mid_flats.extend((flat_frames[hmid_exp_i].cds_frames[0,0], flat_frames[lmid_exp_i].cds_frames[0,1]))
                    
                else:
                    gain_label.append(gain)
                    sat_flats.append(flat_frames[max_i].cds_frames[0])
                    medians = np.array([ff.get_median((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])
                    variance = np.array([ff.get_variance((0,ff.dev_size[0],0,ff.dev_size[1])) for ff in flat_frames])

                    exp_times.append(exp)
                    med_sig.append(medians)
                    var.append(variance)
                    
                    # Find the mid-range exposures
                    mid_exp_i = np.where( (medians > 5000) & (medians < 15000) )[0][0]
                    mid_flats.append(flat_frames[mid_exp_i].cds_frames[0])
                    mid_flat_times.append(exp[mid_exp_i])
                    
            # Mid-range flats
            for i, flat in enumerate(mid_flats):
                mmin, mmax = min(flat.flatten()), max(flat.flatten())
                
                summary_text = f'Gain mode: {gain_label[i]}\n'
                summary_text += f'Illuminated {mid_flat_times[i]}s exposure\n'
                summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU\n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, flat, f'Mid-range flat frame - gain: {gain_label[i]}', summary_text)

            # Well depth report
            for i, flat in enumerate(sat_flats):
                mmin, mmax = min(flat.flatten()), max(flat.flatten())
                n_bad = np.sum(flat < 5000)
                bad_percent = n_bad / flat.size * 100.
                
                summary_text = f'Gain mode: {gain_label[i]}\n'
                summary_text += f'Illuminated {max_exp}s exposure (saturated)\n'
                summary_text += f'Min median pixel value: {mmin} ADU; max median pixel value: {mmax} ADU\n'
                summary_text += f'Pixels < 5000 ADU: {n_bad}, {bad_percent:.3f}%\n'
                
                # Create a standard histogram page and save it to the pdf
                hist_page(pdf, flat, f'Well depth (saturated flat frame) - gain: {gain_label[i]}', summary_text)
                    
            # Linearity and PTC plots
            fig = plt.figure(figsize=[8.5,11],dpi=300)
            plt.suptitle(f'Linearity and photon transfer curves')
            
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
            
            # Linearity curve
            plt.sca(ax1)
            plt.title(f'Linearity')
            for j in range(4): plt.scatter(exp_times[j],med_sig[j],label=f'{gain_label[j]}',marker='x')
            plt.xlabel('Exposure Time (s)')
            plt.ylabel('Median Signal (ADU)')
            plt.loglog()
            plt.legend(fontsize=10,ncol=4,loc=8,bbox_to_anchor=(0.5, -0.2))
                
            # PTC
            plt.sca(ax2)
            plt.title(f'Photon Transfer Curve')
            for j in range(4): plt.scatter(med_sig[j],var[j],label=f'{gain_label[j]}',marker='x')
            plt.xlabel('Median Signal (ADU)')
            plt.ylabel('Variance (ADU^2)')
            plt.loglog()
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
        # TODO: something for guiding rows
        # TODO: line fitting for linearity/PTC plots
        # TODO: bad pixel maps

def hist_page(pdf, data, title, summary_text):
    '''
    Default style of page showing a figure of the image, histograms, and summary text
    '''
    dmin, dmax = min(data.flatten()), max(data.flatten())
    data_range = int(np.ceil(dmax) - np.floor(dmin))

    fig = plt.figure(figsize=[8.5,11],dpi=250)
    plt.suptitle(title)
    
    data_hist, bin_edges = np.histogram(data,bins=100)
    data_hist2, bin_edges2 = np.histogram(data,bins=data_range) # i.e. binsize=1

    # Output a frame image and histograms
    ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((5,2), (2,0), rowspan=2)
    ax3 = plt.subplot2grid((5,2), (2,1), rowspan=2)
                    
    plt.sca(ax1)
    plt.imshow(data, vmin=0, vmax=np.percentile(data,99.9))
    plt.colorbar(label='ADU', shrink=0.9)
    
    plt.sca(ax2)
    plt.stairs(data_hist,edges=bin_edges)
    plt.xlabel('ADU')
    plt.ylabel('Pixel Count')
    plt.text(0.55,0.95,'Whole range',transform=ax2.transAxes)
    plt.semilogy()
    
    plt.sca(ax3)
    plt.stairs(data_hist2,edges=bin_edges2)
    plt.xlabel('ADU')
    plt.xlim(np.percentile(data,[1.e-2,100-1.e-2]))
    plt.text(0.35,0.95,'0.01 - 99.99 percentile',transform=ax3.transAxes)
    plt.semilogy()
                    
    plt.text(0.0, -0.3, summary_text, transform=ax2.transAxes, verticalalignment='top')

    plt.subplots_adjust(hspace=0.25)
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

