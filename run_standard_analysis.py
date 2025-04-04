import cmost_camera

# WARNING: don't put underscores in camid or detid or it will mess up the analysis script

# Define camera setup here
camid = 'cmost'                 # Camera name i.e. 'cmost' or 'cmostjpl'
detid = 'DIE103'                # Device name (see Detector Manifest)
ledw = '800int'                 # LED wavelength in nm
# Supported options are: '255', '260', '285', '310', '340', '800', '800int', 'None'
# (there are two 800nm options, 800/800int for extrenal/internal LEDs)
config_filepath = '/home/user/CMOST/cmost1k1k.cfg' # Absolute path to CameraD config file

# Define which exposures to take here (time for a 1k x 1k detector)
singleframe = True              # Single frames to estimate readout time (<1 min)
noise = True                    # Acquire a noise density spectrum for 3 random pixels (<1 min)
bias = True                     # 100x minimum-length exposures in all gain modes (~5 min)
longdark = True                 # 3 x 3hr darks in hdr mode (~9 hrs)
opdark = True                   # 'Operating mode' darks (~4 hrs)
persist = False                 # Persistence test (not available for all LEDs) (<1 min)
flat = True                     # Flat frame sequence for PTC (~3 hrs)
singleflat, flatv = False, -1   # 20x 300s flats at given LED voltage (~1.5 hrs)

# This commands the camera to take the standard analysis exposures requested
cmost_camera.standard_analysis_exposures(camid,detid,config_filepath,ledw=ledw,singleframe=singleframe,noise=noise,bias=bias,longdark=longdark,opdark=opdark,persist=persist,flat=flat,singleflat=singleflat,flatv=flatv)
