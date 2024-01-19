# cmost-analysis overview

This package contains tools and notebooks for analyzing CMOST images

## How to use

I/O is handled through the Exposure() class in cmost\_exposure.py. This class can:
* read a FITS file
* perform correlated double sampling
* get statistical properties of subframes (regions of interest)

In the future, it will also be able to handle flat-fielding, gain per amplifier etc.

There are also utility functions in cmost\_exposure.py for reading multiple image files into a list of Exposure objects:
* load\_by\_file\_prefix(path\_prefix): loads all FITS files beginning with the filepath string provided
* load\_by\_filepath(filepaths): loads all FITS files from provided list of filepaths
* scan\_headers(directory): returns a table of FITS files and their properties in a given directory

Image data FITS files are to be kept in directory /data, which .gitignore is set up to ignore. On the CMOST computer, this is softlinked to CMOST/Analysis/data. 

Analysis (plotting images, fitting lines etc.) is to be done using Jupyter notebooks in the /notebooks directory. Make a record of files, notebook used, and plots produced in the Data Log on Sharepoint. Create a new notebook per day/experiment, so that we always have old images to hand. Save plots to notebooks/plots. Format for notebook and plot filenames is YYYYMMDD\_DESCRIPTIVE\_EXPERIMENT\_TITLE e.g.:
* 20210122\_LED\_warmup.ipynb
* 20210122\_PTC\_by\_LED\_voltage.pdf

Do not put analysis code in cmost\_exposure.py, and if you find yourself repeatedly doing I/O steps in the notebooks, put it in cmost\_exposure.py. Do not do development on cmost\_exposure.py from the CMOST computer, as it is not set up to push changes back up to Github. Put in an issue here instead. 
