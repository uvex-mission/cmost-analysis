'''
    Utility functions for working with the CMOST detector
    
    NOTE: pyarchon requires Python 2.7. Code must be compatible,
    intialize Python 2.7 by using 'conda activate py2' before running
'''
from __future__ import print_function
import urllib2 as url
from datetime import datetime

def get_temp(cmost=1):
    '''
    Get the device temperature from the detlab site
    
    To-do: implement ability to pass a previous date and time
    
    Parameters
    ----------
    cmost : int
        Which camera is being used. 1 = cmost, 2 = cmost-jpl
        
    Return
    ------
    time : string
        The time of temperature reading, in ISO format
        
    temp : float
        The device temperature in K
    '''
    today = datetime.today()
    year = today.strftime('%Y')
    datestring = today.strftime('%Y%m%d')
    
    # Determine which URL to look at for data
    if cmost == 1:
        temp_url = 'https://sites.astro.caltech.edu/~detlab/cmost/'+year+'/'+datestring+'/temps.csv'
    elif cmost == 2:
        temp_url = 'https://sites.astro.caltech.edu/~detlab/cmost-jpl/'+year+'/'+datestring+'/temps.csv'
    else:
        print('Invalid CMOST number passed to get_temp(): '+str(cmost))
        return today.isoformat(), '-1'

    # Attempt to open temperature data file
    try:
        temps = url.urlopen(temp_url)
    except:
        print('Unable to open '+temp_url)
        return today.isoformat(), '-1'
   
    # Get last line of file
    for line in temps:
        pass
    last_line = line.decode('utf-8').split(',')
    
    time = datetime.strptime(last_line[0],'%m/%d/%y %H:%M:%S')
    temp = last_line[1] # This will not work for dates before ~01/07/2021
    # If implementing past dates, make sure to look at column [2] instead

    return time.isoformat(), temp
