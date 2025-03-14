import cmost_camera

# WARNING: don't put underscores in camid or detid or it will mess up the analysis script

# Define camera setup here
camid = 'cmost'                 # Camera name i.e. 'cmost' or 'cmostjpl'
detid = 'DIE06'                 # Device name (see Detector Manifest)
ledw = '800'                    # LED wavelength in nm
config_filepath = '/home/user/CMOST/cmost1k1k.cfg' # Absolute path to CameraD config file

# Define which exposures to take here (time for a 1k x 1k detector)
singleframe = True              # Single frames to estimate readout time (<1 min)
bias = True                     # 100x minimum-length exposures in all gain modes (~5 min)
longdark = True                 # 3 x 3hr darks in hdr mode (~9 hrs)
opdark = True                   # 'Operating mode' darks (~4 hrs)
flat = True                     # Flat frame sequence for PTC (~3 hrs)
singleflat, flatv = False, -1   # 20x 300s flats at given LED voltage (~1.5 hrs)

# This commands the camera to take the standard analysis exposures requested
cmost_camera.standard_analysis_exposures(camid,detid,config_filepath,ledw=ledw,singleframe=singleframe,bias=bias,longdark=longdark,opdark=opdark,flat=flat,singleflat=singleflat,flatv=flatv)
