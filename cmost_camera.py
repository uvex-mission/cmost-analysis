'''
    Functions relating to taking data with the CMOST camera
    Uses the pyarchon module to control the camera
    
    NOTE: pyarchon requires Python 2.7. Code must be compatible,
    intialize Python 2.7 by using 'conda activate py2' before running
    
    Usage:
    python cmost_camera.py EXPOSURESET CAMID DETID
'''
from __future__ import print_function
import os, sys
sys.path.append('..')
import numpy as np
import time
from datetime import datetime
from pyarchon import cmost as cam
from cmost_utils import get_temp


def dwelltest(camid,detid):
    basename = setup_camera(camid,detid)
    start = time.time()
    print('Taking test NUV guiding dwell')
    exp_UVEX_NUV_dwell(basename+'_NUVguidingdark_test')
    print('Total time elapsed: '+str(time.time() - start)+' s')
    cam.close()


def standard_analysis_exposures(camid,detid):
    '''
    Function to take all exposures required for standard analysis
    to be performed on all chips. DO NOT MODIFY - we need the exposures
    and conditions to be consistent across all devices.
    '''
    start = time.time()
    
    # TODO: Initialize Lakeshore in here to take temperatures
    '''
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
    basename = setup_camera(camid,detid)
    
    # Get device temperature (currently not being recorded)
    if camid == 'cmost': c = 1
    elif camid == 'cmostjpl': c = 2
    else: c = 3
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
    
    # Wait 10 mins for biases to settle
    print('Waiting for biases to settle (10 mins)...')
    time.sleep(600)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Take analysis exposures

    # Minimum-length dark frames - 20 frames for each gain mode
    print('Taking bias frames (~1 min)...')
    cam.key('EXPTIME=0//Exposure time in ms')
    cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image
    cam.key('NORESET=1//Reset frame has been removed')
    for g in ['high','low','hdr']:
        cam.set_basename(basename+'_bias_'+g)
        set_gain(g)
        cam.expose(0,20,0)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Switch to longexposure mode
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('LONGEXPO=1// 1|0 means exposure in s|ms')
    
    # Standard operating mode darks, remain in dual-gain mode
    # FUV mode - 9 + 900s x 9
    print('Taking FUV standard operating mode darks (2.3 hours)...')
    for i in np.arange(9):
        set_gain('hdr')
        cam.set_basename(basename+'_FUVdark'+str(i)+'_9')
        cam.key('EXPTIME=9//Exposure time in s')
        cam.expose(9,1,0)
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
    for i in np.arange(9):
        set_gain('hdr')
        cam.set_basename(basename+'_NUVdark'+str(i)+'_3')
        cam.key('EXPTIME=3//Exposure time in s')
        cam.expose(3,1,0)
        cam.set_basename(basename+'_NUVdark'+str(i)+'_300')
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
    
    # Linearity/PTC illuminated flat field exposures of increasing exposure time
    # Switch on LED
    cam.setled(1.7) #Voltage TBD
    print('Waiting for LED to settle...')
    time.sleep(300) # Wait for LED to settle, 5 min
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    cam.key('LED=1.7//LED voltage in Volts')
    print('Taking illuminated flat field exposures (~4 hours)...')
    for g in ['high','low','hdr']:
        print('Flats for '+g+' gain')
        set_gain(g)
        # Loop through exposure times between 1 and ~1260s
        # Two exposures per exposure time for PTC generation
        for t in np.rint(np.logspace(0,3.1,10)):
            cam.set_basename(basename+'_flat_'+g)
            cam.key('EXPTIME='+str(t)+'//exposure time in seconds')
            cam.expose(int(t),2,0)
        print('Time elapsed: '+str(time.time() - start)+' s')

    # Switch off LED and camera
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    print('Total time elapsed: '+str(time.time() - start)+' s')

    
def long_darks(camid,detid,gain):
    '''
    Function to take long dark exposures to be performed on all chips.
    DO NOT MODIFY - we need the exposures and conditions to be
    consistent across all devices.
    '''
    start = time.time()
    
    # Initialize camera
    basename = setup_camera(camid,detid)
    
    # Get device temperature (currently not being recorded)
    if camid == 'cmost': c = 1
    elif camid == 'cmostjpl': c = 2
    else: c = 3
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
    
    # Wait 10 mins for biases to settle
    print('Waiting for biases to settle (10 mins)...')
    time.sleep(600)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Switch to longexposure mode
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('LONGEXPO=1// 1|0 means exposure in s|ms')
    cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image
    cam.key('NORESET=1//Reset frame has been removed')
    
    # Take 3 hr darks
    g = set_gain(gain)
    if g:
        for i in range(3):
            ind = str(i)
            print('Taking '+gain+' 3-hour dark... ('+str(i+1)+' of 3)')
            cam.set_basename(basename+'_longdark'+ind+'_'+gain)
            cam.key('EXPTIME=10800//Exposure time in seconds')
            cam.expose(10800,1,0)
            print('Taking short darks...')
            cam.set_basename(basename+'_longdark'+ind+'_short'+gain)
            cam.key('EXPTIME=3//Exposure time in seconds')
            cam.expose(3,10,0)
            print('Time elapsed: '+str(time.time() - start)+' s')

    # Switch off LED and camera
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    print('Total time elapsed: '+str(time.time() - start)+' s')
    

def setup_camera(camid,detid):
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
    return True
    
    
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

def exp_UVEX_NUV_HDR(basename,first_exp): # NUX Exposure with guiding
    print('Beginning NUV guiding HDR cycle')
    nuvtime = time.time()
    # Starts with 3s of guiding followed by a low gain readout followed by 300s of guiding followed by and HDR readout
    cam.__send_command('longexposure','false')
    cam.set_param('longexposure',0)
    cam.key('longexposure=0// 1|0 means exposure in s|ms')
    set_gain('high') #Guiding pixels are in high gain mode
    cam.set_mode('GUIDING')
    cam.set_param('BOI_start',200) # First row of the Band Of Interest
    if first_exp:
        cam.set_param('InitFrame',1) # will apply initial reset frame but doesn't not capture the resulting image
    print('NUV short exp setup done, time elapsed: '+str(time.time() - nuvtime)+' s') # Should be negligible
    cam.set_basename(basename+'_UVEXNUV_1_3sguiding_')
    cam.expose(1000,3,0) # 3 Guiding Frames at 1 Hz
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
    cam.expose(1000,300,0) # 300 Guiding Frames
    print('300 guiding frames taken, time elapsed: '+str(time.time() - nuvtime)+' s')
    set_gain('hdr') # Mode to Full Frame HDR Rolling Reset
    cam.set_basename(basename+'_UVEXNUV_4_hdr_')
    cam.expose(0,1,0)
    print('Long frame readout done, total time elapsed: '+str(time.time() - nuvtime)+' s')
    # cam.set_param('InitFrame',1) # InitFrame will be handled in dwell sequence
    

if __name__ == '__main__':
    '''
    Usage:
    python cmost_camera.py EXPOSURESET CAMID DETID
    '''
    print('Executing dwell test')
    dwelltest('cmost','test')
    exit()

    # Get command line parameters
    if len(sys.argv) < 3:
        print('''
Usage:
    python cmost_camera.py standard CAMID DETID
    python cmost_camera.py longdark CAMID DETID GAIN
            ''')
        exit()

    # Pick the exposure set to execute
    if sys.argv[1] == 'standard':
        print('Taking standard exposure set (~8 hours)')
        if len(sys.argv) < 3: camid = raw_input('Camera ID (cmost or cmostjpl): ')
        else: camid = sys.argv[2]
        if len(sys.argv) < 4: detid = raw_input('Detector ID: ')
        else: detid = sys.argv[3]
        standard_analysis_exposures(camid,detid)
    elif sys.argv[1] == 'longdark':
        print('Taking long darks (~9 hours)')
        if len(sys.argv) < 3: camid = raw_input('Camera ID (cmost or cmostjpl): ')
        else: camid = sys.argv[2]
        if len(sys.argv) < 4: detid = raw_input('Detector ID: ')
        else: detid = sys.argv[3]
        if len(sys.argv) < 5: detid = raw_input('Gain (high, low, or hdr): ')
        else: gain = sys.argv[4]
        long_darks(camid,detid,gain)
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
'''
