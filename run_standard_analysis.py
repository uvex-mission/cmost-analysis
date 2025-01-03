import cmost_camera
from pyarchon import cmost as cam

# Define camera setup here
camid = 'cmost'
detid = 'DIE03'
ledw = '800'

# Define which exposures to take here
singleframe = True
bias = True
longdark = True
opdark = True
flat = True
singleflat, flatv = False, -1

# This commands the camera to take the standard analysis exposures requested
cmost_camera.standard_analysis_exposures(camid,detid,ledw=ledw,singleframe=singleframe,bias=bias,longdark=longdark,opdark=opdark,flat=flat,singleflat=singleflat,flatv=flatv)
