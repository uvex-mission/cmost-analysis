# Script for just taking noise measurement and 600 'bias' for long darks
import cmost_camera
from pyarchon import cmost as cam
from cmost_utils import get_temp
from cmost_analysis import write_fits_image
import time

# WARNING: don't put underscores in camid or detid or it will mess up the analysis script

# Define camera setup here
camid = 'cmost'                 # Camera name i.e. 'cmost' or 'cmostjpl'
detid = 'DIEBSI173'                # Device name (see Detector Manifest)
ledw = '800'                 # LED wavelength in nm
# Supported options are: '255', '260', '285', '310', '340', '800', '800int', 'None'
# (there are two 800nm options, 800/800int for external/internal LEDs)

config_filepath = '/home/user/CMOST/cmost4kx2k_20250902.cfg' # Absolute path to CameraD config file

# Set up camera
# Define the output directory and parse config file
output_dir = None
with open(config_filepath) as f:
    for line in f:
        # Get the ACF
        if line.startswith('DEFAULT_FIRMWARE'): acf_file = line.split('=')[1][:-1]
        # Define the output directory
        if line.startswith('IMDIR'):
            data_dir = line.split('=')[1][:-1]
            folderstring = time.strftime('%Y%m%d', time.gmtime())+'_'+detid
            v = np.sum([folderstring in d for d in os.listdir(data_dir)])
            if v > 0: output_dir = data_dir+'/'+folderstring+'_'+str(v)+'/'
            else: output_dir = data_dir+'/'+folderstring+'/'

if output_dir:
    os.mkdir(output_dir)
else:
    print('Problem defining output directory')
    exit()

# Dump .acf and notes file into the output directory
notes_filepath = cmost_camera.dump_info(acf_file, output_dir)

# Start up the camera
basename = cmost_camera.setup_camera(camid, detid, output_dir)

# Get device temperature (currently not being recorded)
temptime, temp = get_temp(cmost=1)
cam.key('TEMP='+str(temp)+'//ZIF Socket temperature in Kelvin')
print('Socket temperature: '+str(temp)+' K')

# Switch to longexposure mode and set InitFrame to true as default
cam.__send_command('longexposure','true')
cam.set_param('longexposure',1)
cam.key('LONGEXPO=1// 1|0 means exposure in s|ms')
cam.set_param('InitFrame',1) # Apply initial reset frame but don't capture resulting image
cam.key('NORESET=1//Reset frame has been removed') # For backwards compatiblity
cam.key('LEDWAVE=.') # Turn off LEDWAVE parameter

# Wait 25 mins for biases to settle after turning on camera
print('Waiting for biases to settle (25 mins)...')
time.sleep(1500)

# Take bias frames for noise measurement
# (parseable by standard analysis script)
print('Taking bias frames (~5 min)...')
for g in ['high','low','hdr']:
    cam.set_basename(basename+'_bias_'+g)
    set_gain(g)
    cam.expose(0,100,0)

# Take a 600s dark in dual-gain mode
# (not parseable by standard analysis script)
print('Taking 600s dark...')
cam.set_basename(basename+'_600dark')
cam.key('EXPTIME=600//Exposure time in seconds')
cam.expose(600,1,0)

# Shut down camera
print('Exposures complete, shutting down camera')
cam.setled(0)
cam.close()

# Since 600s dark isn't parseable by standard analysis script
# Unpack into CDS-applied FITS file here
dark600 = load_by_file_prefix(output_dir+'/'+basename+'_600dark')[0]
dark600_frames, dark600_comment = {}
dark600_frames['high (dual-gain)'] = dark600.cds_frames[0,0]
dark600_frames['low (dual-gain)'] = dark600.cds_frames[0,1]
dark600_comment = {'high (dual-gain)': '600s dark frame, high-gain', 'low (dual-gain)': '600s dark frame, low-gain'}

dark600_outpath = write_fits_image(dark600_frames, 'dark bias frame', dark600_comment, output_dir+'/'+basename+'_600dark_cds.fits')
print('Saved 600s dark CDS frames to '+dark600_outpath)
