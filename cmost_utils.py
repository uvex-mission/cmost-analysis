''' Utility functions for working with the CMOST detector '''

import urllib2 as url
import os
from datetime import datetime

# Function to retrieve the latest temperature
# To-do later if needed: implement ability to pass a previous date and time and retrieve closest temperature
def get_temp(cmost=1):
    
    today = datetime.today()
    year = today.strftime('%Y')
    datestring = today.strftime('%Y%m%d')
    
    # Determine which URL to look at for data
    if cmost == 1:
        temp_url = 'https://sites.astro.caltech.edu/~detlab/cmost/'+year+'/'+datestring+'/temps.csv'
    elif cmost == 2:
        temp_url = 'https://sites.astro.caltech.edu/~detlab/cmost-jpl/'+year+'/'+datestring+'/temps.csv'
    else:
        print 'Invalid CMOST number passed to get_temp(): '+str(cmost)
        return today.isoformat(), '-1'

    # Attempt to open temperature data file
    try:
        temps = url.urlopen(temp_url)
    except:
        print 'Unable to open '+temp_url
        return today.isoformat(), '-1'
   
    # Get last line of file
    for line in temps:
        pass
    last_line = line.decode('utf-8').split(',')
    
    time = datetime.strptime(last_line[0],'%m/%d/%y %H:%M:%S')
    temp = last_line[1] # This will not work for dates before ~01/07/2021
    # If implementing past dates, make sure to look at column [2] instead

    return time.isoformat(), temp
