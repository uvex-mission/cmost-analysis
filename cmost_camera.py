'''
    Functions relating to taking data with the CMOST camera
    Uses the pyarchon module to control the camera
    
    NOTE: pyarchon requires Python 2.7. Code must be compatible,
    intialize Python 2.7 by using 'conda activate py2' before running
    
    Usage:
    python cmost_camera.py EXPOSURESET CAMID DETID LED
    
    For a more tailored exposure set, call standard_analysis_exposures() from a script
'''
from __future__ import print_function
import os, sys, shutil
sys.path.append('..')
import numpy as np
import time
from datetime import datetime
from pyarchon import cmost as cam
from cmost_utils import get_temp
import subprocess

def dwelltest(camid,detid):
    basename = setup_camera(camid,detid)
    start = time.time()
    print('Taking test NUV guiding dwell')
    exp_UVEX_NUV_dwell(basename+'_NUVguidingdark_test')
    print('Total time elapsed: '+str(time.time() - start)+' s')
    cam.close()


def standard_analysis_exposures(camid, detid, config_filepath, ledw='None', singleframe=True, bias=True, longdark=True, opdark=True, flat=True, singleflat=False, flatv=-1):
    '''
    Function to take all exposures required for standard analysis
    to be performed on all chips. By default takes all exposure sets,
    but this can be modified depending on the flags that are set. Each
    exposure set can be run independently.
    
    - singleframe: A minimum-length single frame readout in each gain mode,
            for estimating the readout time
    - bias: 100x minimum-length exposures in each gain mode, for bias frame
            and read noise calculation
    - longdark: 3x 3-hour dark exposures in dual-gain mode, for dark current
            measurement and glow characterization
    - opdark: 9 sets of exposures in FUV mode (9+900s), NUV mode (3+300s) and
            NUV guiding mode (3+300s with 10Hz guiding)
    - flat: Sets of flat frames for PTC generation (requires ledw to be set)
    - singleflat: 20x flat frame in a particular configuration (requires ledw and flatv to be set)
    '''
    start = time.time()
    
    # Define the output directory
    # Parse config file
    output_dir = None
    with open(config_filepath) as f:
        for line in f:
            # Get the ACF
            if line.startswith('DEFAULT_FIRMWARE'): acf_file = line.split('=')[1][:-1]
            # Define the output directory
            if line.startswith('IMDIR'):
                data_dir = line.split('=')[1][:-1]
                datestring = time.strftime('%Y%m%d', time.gmtime())
                v = np.sum([datestring in d for d in os.listdir(data_dir)])
                if v > 0: output_dir = data_dir+'/'+datestring+'_'+str(v)+'/'
                else: output_dir = data_dir+'/'+datestring+'/'
    
    if output_dir:
        os.mkdir(output_dir)
    else:
        print('Problem defining output directory')
        exit()
    
    # Dump .acf and notes file into the output directory
    notes_filepath = dump_info(acf_file, output_dir)

    # Start up the camera
    basename = setup_camera(camid, detid, output_dir)
    
    # Get device temperature (currently not being recorded)
    if camid == 'cmost': c = 1
    elif camid == 'cmostjpl': c = 2
    else: c = 3
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
    
    # Switch to longexposure mode and set InitFrame to true as default
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('LONGEXPO=1// 1|0 means exposure in s|ms')
    cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image
    cam.key('NORESET=1//Reset frame has been removed') # For backwards compatiblity
    
    # Turn off LEDWAVE parameter until needed
    cam.key('LEDWAVE=.')
    
    # Wait 25 mins for biases to settle after turning on camera
    print('Waiting for biases to settle (25 mins)...')
    time.sleep(1500)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Take analysis exposures
    
    # Single minimum-length frame in each gain mode to get readout timing
    if singleframe:
        print('Getting readout times (<1 min)...')
        cam.key('EXPTIME=0//Exposure time in ms')
        time_notes = 'Readout times:\n'
        for g in ['high','low','hdr']:
            cam.set_basename(basename+'_singleframe_'+g)
            set_gain(g)
            exp_start = time.time()
            cam.expose(0,1,0) # Hack that incorporates FITS write time, only an approximation
            readout_time = time.time() - exp_start
            time_notes += g+': '+str(readout_time)+'\n'
        # Output time notes into the notes file
        notes_file = open(notes_filepath,'a')
        notes_file.write(time_notes)
        notes_file.close()
        print('Time elapsed: '+str(time.time() - start)+' s')

    # Minimum-length dark frames - 100 frames for each gain mode
    if bias:
        print('Taking bias frames (~5 min)...')
        for g in ['high','low','hdr']:
            cam.set_basename(basename+'_bias_'+g)
            set_gain(g)
            cam.expose(0,100,0)
        print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Take 3 x 3 hr darks in HDR mode only
    if longdark:
        print('Taking long darks (9 hours)...')
        set_gain('hdr')
        for i in range(3):
            ind = str(i)
            print('Taking hdr 3-hour dark... ('+str(i+1)+' of 3)')
            cam.set_basename(basename+'_longdark'+ind+'_hdr')
            cam.key('EXPTIME=10800//Exposure time in seconds')
            cam.expose(10800,1,0)
            print('Time elapsed: '+str(time.time() - start)+' s')
            
            # Get temperature again since some time has passed
            temptime, temp = get_temp(cmost=c)
            cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
            print('Socket temperature: '+str(temp)+' K')
            
        # Also take a 10-minute dark for subtraction
        cam.set_basename(basename+'_longdark'+ind+'_shorthdr')
        cam.key('EXPTIME=600//Exposure time in seconds')
        cam.expose(600,1,0)
        print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Standard operating mode darks, remain in dual-gain mode
    if opdark:
        set_gain('hdr')
        # FUV mode - 9 + 900s x 9
        print('Taking FUV standard operating mode darks (2.3 hours)...')
        for i in np.arange(9):
            cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image #Added by Tim
            cam.set_basename(basename+'_FUVdark'+str(i)+'_9')
            cam.key('EXPTIME=9//Exposure time in s')
            cam.expose(9,1,0)

            cam.set_param('InitFrame',0) # Added by Tim
            cam.set_basename(basename+'_FUVdark'+str(i)+'_900')
            cam.key('EXPTIME=900//Exposure time in s')
            cam.expose(900,1,0)
            print('Time elapsed: '+str(time.time() - start)+' s')
        
        # Get temperature again since some time has passed
        temptime, temp = get_temp(cmost=c)
        cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
        print('Socket temperature: '+str(temp)+' K')
        
        # NUV mode - 3 + 300s x 9
        print('Taking NUV standard operating mode darks (45 mins)...')
        set_gain('hdr')
        for i in np.arange(3):
            cam.set_param('InitFrame',1) # Added by Tim
            cam.set_basename(basename+'_NUVdark'+str(i*3)+'_3')
            cam.key('EXPTIME=3//Exposure time in s')
            cam.expose(3,1,0)
            cam.set_param('InitFrame',0) # Added by Tim
            cam.set_basename(basename+'_NUVdark'+str(i*3)+'_300')
            cam.key('EXPTIME=300//Exposure time in s')
            cam.expose(300,1,0)
            cam.set_basename(basename+'_NUVdark'+str(i*3 + 1)+'_3')
            cam.key('EXPTIME=3//Exposure time in s')
            cam.expose(3,1,0)
            cam.set_basename(basename+'_NUVdark'+str(i*3 + 1)+'_300')
            cam.key('EXPTIME=300//Exposure time in s')
            cam.expose(300,1,0)
            cam.set_basename(basename+'_NUVdark'+str(i*3 + 2)+'_3')
            cam.key('EXPTIME=3//Exposure time in s')
            cam.expose(3,1,0)
            cam.set_basename(basename+'_NUVdark'+str(i*3 + 2)+'_300')
            cam.key('EXPTIME=300//Exposure time in s')
            cam.expose(300,1,0)
            print('Time elapsed: '+str(time.time() - start)+' s')

        # Get temperature again
        temptime, temp = get_temp(cmost=c)
        cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
        print('Socket temperature: '+str(temp)+' K')
    
        # NUV mode - 300s + 3s, with one 1 Hz guide window
        # 1 Hz -> guide period = 1000 ms
        # Default 10-row band of interest at default location (row 200)
        print('Taking NUV standard operating mode darks with 1 Hz guiding (45 mins)...')
        for i in np.arange(3):
            # Each dwell is 3 cycles of 3+300s
            exp_UVEX_NUV_dwell(basename+'_NUVguidingdark'+str(i))
            #gbasename = basename+'_NUVguidingdark'+str(i)+'_3'
            #take_guiding_exposure(3,'hdr',gbasename,boi_start=200,boi_size=10)
            #gbasename = basename+'_NUVguidingdark'+str(i)+'_300'
            #take_guiding_exposure(300,'hdr',gbasename,boi_start=200,boi_size=10)
            print('Time elapsed: '+str(time.time() - start)+' s')
        
        # Get temperature again
        temptime, temp = get_temp(cmost=c)
        cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
        print('Socket temperature: '+str(temp)+' K')
    
        # Switch back to longexposure mode from guiding
        cam.__send_command('longexposure','true')
        cam.set_param('longexposure',1)
        cam.key('LONGEXPO=1// 1|0 means exposure in s|ms')
        cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image
        cam.key('NORESET=1//Reset frame has been removed')
    
    # Illuminated flat frame sequence for PTC
    if flat:
        if ledw == 'None':
            print('No LED wavelength specified, skipping flats')
        else:
            # Get the ideal voltage range for this LED
            cam.key('LEDWAVE='+str(ledw)+'// LED wavelength in nm')
            voltages = get_ptc_setup(ledw)
            
            if voltages is not None:
                # Switch on LED
                print('Taking illuminated flat field exposures (~3 hours)...')
                for voltage in voltages:
                    cam.setled(voltage)
                    cam.key('LED='+str(voltage)+'// LED voltage in Volts')
                    time.sleep(60)
                    
                    for g in ['high','low','hdr']:
                        print('Flats for '+g+' gain')
                        set_gain(g)
                        # Min-length exposures
                        cam.set_basename(basename+'_flat_'+g+'_'+str(voltage))
                        cam.key('EXPTIME=0//exposure time in seconds')
                        cam.expose(0,2,0)
                        # Loop through exposure times between 1 and ~200s
                        # Two exposures per exposure time for PTC generation
                        for t in np.rint(np.logspace(0,2.3,10)):
                            cam.key('EXPTIME='+str(t)+'//exposure time in seconds')
                            cam.expose(int(t),2,0)
                        print('Time elapsed: '+str(time.time() - start)+' s')
                    
                    # Get temperature again
                    temptime, temp = get_temp(cmost=c)
                    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
                    print('Socket temperature: '+str(temp)+' K')
            else:
                print('No voltage set found for this LED, skipping flats')
            
    # Set of illuminated flats with single setup
    if singleflat:
        if ledw == 'None':
            print('No LED wavelength specified, skipping flats')
        elif flatv < 0:
            print('No LED voltage specified, skipping flats')
        else:
            cam.key('LEDWAVE='+str(ledw)+'// LED wavelength in nm')
            gain = 'high'
            t = 300
            # Switch on LED
            cam.setled(flatv)
            cam.key('LED='+str(flatv)+'// LED voltage in Volts')
            print('Waiting for LED to settle...')
            time.sleep(60) # Wait for LED to settle, 1 min
            print('Time elapsed: '+str(time.time() - start)+' s')
            
            set_gain(gain)
            cam.set_basename(basename+'_singleflat_'+gain+'_'+str(flatv))
            cam.key('EXPTIME='+str(t)+'//exposure time in seconds')
            cam.expose(int(t),20,0)
            print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Switch off LED and camera
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    
    print('Total time elapsed: '+str(time.time() - start)+' s')

def setup_camera(camid,detid,output_dir=None):
    '''
    Initialize camera and return the basename
    '''
    print('Initializing camera...')
    cam.open()
    cam.__send_command('datacube', 'true')
    
    # Set device-specific keywords
    cam.key('CAMERAID='+camid+'//Camera ID')
    cam.key('DETID='+detid+'//Device ID')
    
    # Set filename base
    basename = camid+'_'+detid
    cam.set_basename(basename)
    
    # Start with LED forced off
    cam.setled(-0.1)
    cam.key('LED=-0.1//LED voltage in Volts')
    print('LED off')
   
    # Activate Clocks
    cam.set_param('Start',1)
    
    if output_dir:
        # Set and fix the output directory to provided output directory
        cam.__send_command('autodir', 'no')
        cam.__send_command('imdir', output_dir)
    # Otherwise CameraD will simply output into dated folder as usual
 
    return basename

def set_gain(gain):
    '''
    Set camera gain and header
    
    Parameters
    ----------
    gain : string
        Gain mode. Can be 'high', 'low', 'hdr'
    '''
    if gain == 'high':
        cam.set_mode('ROLLINGRESET')
        cam.set_param('GainHigh',1)
        cam.set_param('GainLow',0)
    elif gain == 'low':
        cam.set_mode('ROLLINGRESET')
        cam.set_param('GainHigh',0)
        cam.set_param('GainLow',1)
    elif gain == 'hdr':
        cam.set_mode('ROLLINGRESET_HDR')
        # Note: gain parameters are ignored in HDR mode
        # But we'll set them to 0 here for record-keeping
        cam.set_param('GainHigh',0)
        cam.set_param('GainLow',0)
    else:
        print('Invalid gain')
        return False
    
    cam.key('GAIN='+gain)
    cam.__send_command('longexposure','true')
    return True

def dump_info(acf_file, output_dir):
    '''
    Take information from config file and dump acf and notes file into output directory
    
    Parameters
    ----------
    acf_file : string
        Path to acf file being used with the camera server
    output_dir : string
        Path to the output data directory
    '''
    # Dump acf file into output directory
    shutil.copy2(acf_file,output_dir)
    
    # Create a notes file in output directory
    notes = 'Exposure set taken: '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n\n'
    notes += 'Notes:\n\n'
    
    notes_filepath = output_dir+'/analysis_notes.txt'
    notes_file = open(notes_filepath,'a')
    notes_file.write(notes)
    notes_file.close()
    
    return notes_filepath

def get_ptc_setup(ledw):
    '''
    Get an ideal set of voltages for generating a PTC for the given LED
    Assuming a log-spaced set of exposure times between 1 and 200s
    '''
    switch = { # Calibrated for HfO - setup as of Nov 2024
        '255': [4.0, 7.0, 9.0, 12.0], #[4.5, 5.4, 6.2, 7.0],
        '260': [4.55, 5.0, 7.5, 9.5], #[4.5, 4.6, 4.85, 5.1],
        '285': [4.48, 4.8, 5.2, 5.5], #[4.35, 4.5, 4.6, 4.75],
        '310': [3.8, 4.2, 5.4, 6.6], #[3.7, 3.9, 4.1, 4.3],
        '340': [3.5, 4.0, 4.8, 5.6], #[3.42, 3.56, 3.70, 3.86],
#        '372': [],
        '800': [1.55, 1.62, 1.68, 1.75] #[1.5, 1.55, 1.6, 1.65]
    }
    
    return switch.get(ledw,None)

    
def take_guiding_exposure(t,gain,basename,boi_start=200,boi_size=10):
    '''
    Take an exposure with a 1Hz guiding row activated
    
    Parameters
    ----------
    t : int
        Exposure time in seconds
        
    gain : string
        Gain mode of the full frame readout, can be 'high', 'low', 'hdr'
    
    basename : string
        Basename of output file
        
    boi_start : int
        First row of the guiding band of interest
        Defaults to 200
    
    boi_size : int
        Number of rows of the guiding band of interest
        Defaults to 10 rows
    '''
    # Set guiding row parameters
    cam.set_param('BOI_start',boi_start)
    #cam.set_param('RowGuiding',boi_size)
    
    # Start with guiding readouts equal to t while the rest of the detector is exposing
    cam.__send_command('longexposure','false')
    cam.set_param('longexposure',0)
    cam.key('LONGEXPO=0// 1|0 means exposure in s|ms')
    set_gain('high') # Guiding pixels are read in high gain mode
    cam.set_mode('GUIDING')
    cam.key('NORESET=1//Reset frame has been removed')
    cam.set_param('InitFrame',1)
    cam.set_basename(basename+'_guideframes')
    cam.expose(1000,t,0) # Guiding frames at 1 Hz / 1000ms
    cam.set_param('InitFrame',0) # Do not reset the frame
    set_gain(gain)
    cam.set_basename(basename)
    cam.key('EXPTIME='+str(t*1000)+'//Exposure time in milliseconds')
    cam.expose(0,1,0) # Zero-length exposure since it's been exposing throughout the guiding

'''
Description:    Function to take a full 900s(+9s) dwell sequnece for NUV with guiding.
                Simply three NUV exposures in a row. HDR frame for short exposure.
                Extra logic to ensure InitFrame is only taken and trashed once.

Inputs:
    basename:   exposure_file_name
    first_exp:  flag to identify if InitFrame is required or not (only needed
                for initial frame)
'''
def exp_UVEX_NUV_dwell(basename): 
    # Three exp_UVEX_NUV exposures with guiding for 900 second dwell
    for i in np.arange(3):
        if i == 1:
            first_exp = True
        else:
            first_exp = False
        exp_UVEX_NUV_HDR(basename+"_FullDwell_exp"+str(i),first_exp)
    cam.set_param('InitFrame',1) # End of dwell sequence; enable InitFrame

def exp_UVEX_NUV_HDR(basename,first_exp): # NUV Exposure with guiding
    print('Beginning NUV guiding HDR cycle')
    nuvtime = time.time()
    # Starts with 3s of guiding followed by a low gain readout followed by 300s of guiding followed by and HDR readout
    set_gain('high') #Guiding pixels are in high gain mode
    cam.set_mode('GUIDING')
    cam.set_param('BOI_start',200) # First row of the Band Of Interest
    if first_exp:
        cam.set_param('InitFrame',1) # will apply initial reset frame but doesn't not capture the resulting image
    print('NUV short exp setup done, time elapsed: '+str(time.time() - nuvtime)+' s') # Should be negligible
    cam.set_basename(basename+'_UVEXNUV_1_3sguiding_')
    cam.expose(1,3,0) # 3 Guiding Frames at 1 Hz
    print('3 guiding frames taken, time elapsed: '+str(time.time() - nuvtime)+' s')
    if first_exp:
        cam.set_param('InitFrame',0) # we do not want a reset frame before readout here.
    set_gain('hdr') # Mode to Full Frame HDR Rolling Reset
    cam.set_basename(basename+'_UVEXNUV_2_hdr_short_')
    cam.expose(0,1,0) # Low Gain Frame
    print('Short frame readout done, time elapsed: '+str(time.time() - nuvtime)+' s')
    set_gain('high')
    cam.set_mode('GUIDING') #
    cam.set_basename(basename+'_UVEXNUV_3_300sguiding_')
    cam.expose(1,300,0) # 300 Guiding Frames
    print('300 guiding frames taken, time elapsed: '+str(time.time() - nuvtime)+' s')
    set_gain('hdr') # Mode to Full Frame HDR Rolling Reset
    cam.set_basename(basename+'_UVEXNUV_4_hdr_')
    cam.expose(0,1,0)
    print('Long frame readout done, total time elapsed: '+str(time.time() - nuvtime)+' s')
    # cam.set_param('InitFrame',1) # InitFrame will be handled in dwell sequence
    

if __name__ == '__main__':
    '''
    Usage:
    python cmost_camera.py EXPOSURESET CAMID DETID LED
    '''

    # Get command line parameters
    if len(sys.argv) < 3:
        print('''
Usage:
    python cmost_camera.py standard CAMID DETID CONFIG LED
    python cmost_camera.py longdark CAMID DETID CONFIG
            ''')
        exit()

    # Pick the exposure set to execute
    if sys.argv[1] == 'standard':
        print('Taking standard exposure set (~18 hours)')
        if len(sys.argv) < 3: camid = raw_input('Camera ID (cmost or cmostjpl): ')
        else: camid = sys.argv[2]
        if len(sys.argv) < 4: detid = raw_input('Detector ID: ')
        else: detid = sys.argv[3]
        if len(sys.argv) < 5: config_file = raw_input('Path to config file: ')
        else: config_file = sys.argv[4]
        if len(sys.argv) < 6: ledw = raw_input('LED wavelength (nm): ')
        else: ledw = sys.argv[5]
        
        # Check LED wavelength is one we expect
        if ledw not in ['255', '260', '285', '310', '340', '372', '800', 'None']:
            print('LED wavelength options: 255, 260, 285, 310, 340, 372, 800, None. If None, no flats will be taken. Make sure the correct LED is selected on the camera before running this script!')
            exit()
        
        standard_analysis_exposures(camid,detid,config_file,ledw=ledw)
    elif sys.argv[1] == 'longdark':
        print('Taking long darks (~9 hours)')
        if len(sys.argv) < 3: camid = raw_input('Camera ID (cmost or cmostjpl): ')
        else: camid = sys.argv[2]
        if len(sys.argv) < 4: detid = raw_input('Detector ID: ')
        else: detid = sys.argv[3]
        if len(sys.argv) < 5: config_file = raw_input('Path to config file: ')
        else: config_file = sys.argv[4]

        standard_analysis_exposures(camid,detid,config_file,singleframe=False,bias=False,opdark=False,flat=False)
    else:
        print('Unknown exposure set. Options: standard, longdark')
        exit()



''' Old test functions:

def exp_UVEX_FUV(basename,nexp): # FUV Exposure
    set_gain('high')
    cam.set_param('InitFrame',1) #Apply initial reset frame but doesn't not capture the resulting image
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('longexposure=1// 1|0 means exposure in s|ms')
    t = 900 # 900s FUV exposure
    cam.set_basename(basename+'_UVEXFUV_high_')
    cam.expose(t,nexp,0)

def exp_UVEX_NUV(basename): # NUX Exposure with guiding
    # Starts with 3s of guiding followed by a low gain readout followed by 300s of guiding followed by and HDR readout
    cam.__send_command('longexposure','false')
    cam.set_param('longexposure',0)
    cam.key('longexposure=0// 1|0 means exposure in s|ms')
    set_gain('high') #Guiding pixels are in high gain mode
    cam.set_mode('GUIDING')
    cam.set_param('BOI_start',200) # First row of the Band Of Interest
    cam.set_param('InitFrame',1) # will apply initial reset frame but doesn't not capture the resulting image
    cam.set_basename(basename+'_UVEXNUV_1_3sguiding_')
    cam.expose(1000,3,0) # 3 Guiding Frames at 1 Hz
    cam.set_param('InitFrame',0) # we do not want a reset frame before readout here.
    set_gain('low') # Mode to Full Frame Low Gain Rolling Reset
    cam.set_basename(basename+'_UVEXNUV_2_low_')
    cam.expose(0,1,0) # Low Gain Frame
    set_gain('high')
    cam.set_mode('GUIDING') #
    cam.set_basename(basename+'_UVEXNUV_3_300sguiding_')
    cam.expose(1000,300,0) # 300 Guiding Frames
    set_gain('hdr') # Mode to Full Frame Low Gain Rolling Reset
    cam.set_basename(basename+'_UVEXNUV_4_hdr_')
    cam.expose(0,1,0)
    cam.set_param('InitFrame',1)
    
def test_UVEX_Exposure(camid,detid):
    start = time.time()
    basename = setup_camera(camid,detid)
    time.sleep(300)
    for e in range(20):
        exp_UVEX_NUV(basename)
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    print('Total time elapsed: '+str(time.time() - start)+' s')

def test_PTC(camid,detid):
    start = time.time()
    basename = setup_camera(camid,detid)
    time.sleep(60)
    cam.set_param('longexposure',0)
    cam.key('longexposure=0// 1|0 means exposure in s|ms')
    cam.set_param('InitFrame',1) # will apply initial reset frame but doesn't not capture the resulting image
    cam.setled(1.7) #Voltage TBD
    time.sleep(60) # Wait for LED to settle, 5 min
    cam.key('LED=1.7//LED voltage in Volts')
    print('Taking illuminated flat field exposures to plot PTC...')
    for g in ['high','low','hdr']:
        set_gain(g)
        cam.set_basename(basename+'_ptc_'+g+'_')
        # Loop through exposure times between 1 and ~100s
        for t in 1000*np.rint(np.logspace(0,2,8)).astype(np.int32):
            cam.key('EXPTIME='+str(t)+'//exposure time in seconds')
            cam.expose(t,2,0)

    # Switch off LED and camera
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    print('Total time elapsed: '+str(time.time() - start)+' s')
    
    # Lakeshore connection code for future reference
    try:
        import telnetlib
        host = "coots1.caltech.edu"
        port = 10001
        timeout = 100
        #session = telnetlib.Telnet(host,port,timeout)
        #session.write(b"SETP 1,140")
    except:
        print('Could not connect to Lakeshore controller, no temperature information available')
'''
