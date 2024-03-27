'''
    Script to create a manifest of available CMOST data
    
    Usage: python cmost_manifest.py [DATA_DIR] [PATH_TO_FILE]
'''
import os, sys
sys.path.append('..')
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from cmost_exposure import scan_headers

def create_manifest(data_dir, filename):
    # Load manifest file
    if os.path.isfile(filename):
        m = fits.open(filename)
        manifest = Table(m[1].data)
        directories = manifest['DIRECTORY']
        m.close()
    else:
        directories = []
        manifest = Table()

    # Loop through all available directories
    for dir in os.scandir(data_dir):
        if dir.is_dir():
            # Check if this directory is registered in the manifest
            # If so, skip
            if dir.name in directories:
                continue
            
            # Scan FITS file headers
            header_table = scan_headers(data_dir+'/'+dir.name)
            
            # scan_headers() returns False if directory contains no files
            if header_table:
                # Handle numeric detector IDs
                header_table['DETID'] = header_table['DETID'].astype(str)
                header_table.add_column([dir.name]*len(header_table),name='DIRECTORY')
                manifest = vstack([manifest,header_table])
    
    manifest.write(filename, overwrite=True)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        data_dir = input('Data directory: ')
        filename = input('Manifest filename: ')
    else:
        data_dir = sys.argv[1]
        filename = sys.argv[2]
    
    create_manifest(data_dir, filename)
