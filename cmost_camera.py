'''
    Functions relating to taking data with the CMOST camera
    Uses the pyarchon module to control the camera
    
    NOTE: pyarchon requires Python 2.7. Code must be compatible,
    intialize Python 2.7 by using 'conda activate py2' before running
    
    Run inside the directory where you want the data to be saved
    (i.e. in CMOST/Analysis/data/[your directory])
'''
from __future__ import print_function
import os, sys
sys.path.append('..')
import numpy as np
import time
from datetime import datetime
from pyarchon import cmost as cam
from cmost_utils import get_temp

def test_run(camid,detid):
    ''' For testing only '''
    start = time.time()
    basename = setup_camera(camid,detid)
    time.sleep(600)
    
    print('Taking test frames...')
    cam.key('EXPTIME=0//Exposure time in ms')
    cam.set_basename(basename+'_bias_test')
    set_gain('high')
    cam.expose(0,3,0)
    
    print('Exposures complete, shutting down camera')
    cam.setled(0)
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
    
    # Take analysis exposures

    # Minimum-length dark frames - 20 frames for each gain mode
    print('Taking bias frames (1 min)...')
    cam.key('EXPTIME=0//Exposure time in ms')
    cam.set_basename(basename+'_bias_high')
    set_gain('high')
    cam.expose(0,20,0)
    cam.set_basename(basename+'_bias_low')
    set_gain('low')
    cam.expose(0,20,0)
    # Switch to dual-gain mode
    cam.set_basename(basename+'_bias_hdr')
    set_gain('hdr')
    cam.expose(0,20,0)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Switch to longexposure mode
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('longexposure=1// 1|0 means exposure in s|ms')
    
    # Standard operating mode darks, remain in dual-gain mode
    # FUV mode - 900s + 9s
    print('Taking FUV standard operating mode darks (2.5 hours)...')
    for i in np.arange(10):
        set_gain('low')
        cam.set_basename(basename+'_dark'+i+'_9')
        cam.key('EXPTIME=9//Exposure time in s')
        cam.expose(9,1,0)
        set_gain('hdr')
        cam.set_basename(basename+'_dark'+i+'_900')
        cam.key('EXPTIME=900//Exposure time in s')
        cam.expose(900,1,0)
    print('Time elapsed: '+str(time.time() - start)+' s')
    
    # Get temperature again since some time has passed
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
        
    # NUV mode - 300s + 3s
    print('Taking NUV standard operating mode darks (1 hour)...')
    for i in np.arange(10):
        set_gain('low')
        cam.set_basename(basename+'_dark'+i+'_3')
        cam.key('EXPTIME=3//Exposure time in s')
        cam.expose(3,1,0)
        set_gain('hdr')
        cam.set_basename(basename+'_dark'+i+'_300')
        cam.key('EXPTIME=300//Exposure time in s')
        cam.expose(300,1,0)
    print('Time elapsed: '+str(time.time() - start)+' s')

    # Get temperature again
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
    
    # NUV mode - 300s + 3s, with one 10 Hz guide window
    # 10 Hz -> guide period = 100 ms
    print('Taking NUV standard operating mode darks with 10 Hz guiding (1 hour)...')
    cam.key('guiding_period=100//guiding period in ms')
    cam.set_param('guiding_period',100)
    cam.set_param('RowGuiding',5) # Number of guiding rows to readout
    for i in np.arange(10):
        set_gain('low')
        cam.set_basename(basename+'_darkguiding'+i+'_3')
        cam.key('EXPTIME=3//Exposure time in s')
        cam.set_param('guiding',30)
        cam.expose(3,1,0)
        set_gain('hdr')
        cam.set_basename(basename+'_darkguiding'+i+'_300')
        cam.key('EXPTIME=300//Exposure time in s')
        cam.set_param('guiding',3000)
        cam.expose(300,1,0)
    print('Time elapsed: '+str(time.time() - start)+' s')

    # Get temperature again
    temptime, temp = get_temp(cmost=c)
    cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
    print('Socket temperature: '+str(temp)+' K')
    
    # Linearity/PTC illuminated flat field exposures of increasing exposure time
    # Switch on LED
    cam.setled(1.7)
    cam.key('LED=1.7//LED voltage in Volts')
    print('Taking illuminated flat field exposures (1 hour)...')
    for g in ['high','low','hdr']:
        set_gain(g)
        # Loop through exposure times between 1 and ~500s
        for t in np.rint(np.logspace(0,2.7,10)):
            cam.key('EXPTIME='+str(t)+'//exposure time in seconds')
            cam.expose(t,1,0)

    # Switch off LED and camera
    print('Exposures complete, shutting down camera')
    cam.setled(0)
    cam.close()
    print('Total time elapsed: '+str(time.time() - start)+' s')
    
    # Other notes:
    # use "KEY=." (i.e. set equal to a period) to remove the keyword "KEY"

    
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
    
    # Switch to longexposure mode
    cam.__send_command('longexposure','true')
    cam.set_param('longexposure',1)
    cam.key('longexposure=1// 1|0 means exposure in s|ms')
    
    # Take 3 hr darks
    g = set_gain(gain)
    if g:
        for i in range(3):
            ind = str(i)
            print('Taking '+gain+' 3-hour dark... ('+ind+' of 3)')
            cam.set_basename(basename+'_longdark'+ind+'_'+gain)
            cam.key('EXPTIME=10800//Exposure time in s')
            cam.expose(10800,1,0)
            print('Taking short darks...')
            cam.set_basename(basename+'_longdark'+ind+'_short'+gain)
            cam.key('EXPTIME=3//Exposure time in s')
            cam.expose(3,10,0)

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
    
    # Set pixel hold duration
    cam.set_param('TPixel_hold',860)
    cam.key('TPIXEL_H=860')
    cam.cds('SHD1',860)
    cam.cds('SHD2',860)
    cam.cds('SHP1',790)
    cam.cds('SHP2',800)
    
    # Start with LED forced off
    cam.setled(-0.1)
    cam.key('LED=-0.1//LED voltage in Volts')
    print('LED off')
    
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

if __name__ == '__main__':
    '''
    Usage:
    python cmost_camera.py EXPOSURESET CAMID DETID
    '''
    camid = raw_input('Camera ID (cmost or cmostjpl): ')
    detid = raw_input('Detector ID: ')
    test_run(camid,detid)
    
    '''
    # Get command line parameters
    if len(sys.argv) < 2:
        print('
Usage:
    python cmost_camera.py standard CAMID DETID
    python cmost_camera.py longdark CAMID DETID GAIN
            ')
        exit()
        
    # Make and move in to correct directory

    # Pick the exposure set to execute
    if sys.argv[1] == 'standard':
        print('Taking standard exposure set (~4.5 hours)')
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
        if len(sys.argv) < 5: detid = raw_input('Gain: ')
        else: gain = sys.argv[4]
        long_darks(camid,detid,gain)
    else:
        print('Unknown exposure set. Options: standard, longdark')
        exit()
    

    # Ultralong darks (12 hr, one gain mode at a time)
    # Temperature-dependent darks?
    # PPL 'standard' set?
    '''

