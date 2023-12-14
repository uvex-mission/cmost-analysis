###VERSION 2###
import sys
sys.path.append('..')
from cmost_exposure import load_by_file_prefix, load_by_filepath, scan_headers
import copy 
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import random
from scipy import ndimage
from scipy import signal
from astropy.io import fits
#Library created to generate simulated flat frames with specified parameters
#Based on the formalism outlined in Choi & Hirata "Brighter-fatter Effect in Near-infrared Detectors"

def simFlatField(avg_intensity=10,intensity_gradient_x=[],intensity_gradient_y=[],gradient_offset_x=0,gradient_offset_y=0, sat_charges=1000, gain=1, size=(128,128), t_int=1, dt=0.1, bad_pixel_map=[], hot_pixel_map=[], a_BFE=np.zeros((3,3)), K_IPC=np.array([[0,0,0],[0,1,0],[0,0,0]]), B=0 ,QE=1,readout=1,reset_charge=50,bias_charge=50,plot=False,hot_factor=10,save_fits=True,filename=""):
    #intensity ==> e-/s
    #intensity_gradient ==> float array of polynomial coefficients
    #sat_charges ==> e-
    #gain ==> e-/DN
    #size ==> (row pixels, col pixels)
    #t_int ==> s
    #dt ==> size of time steps
    #bad_pixel_map ==> boolean matrix, 0 = bad pixel, 1 = good pixel
    #hot_pixel_map ==> boolean matrix, 1 = hot pixel, 0 = good pixel
    #B (nonlinearity) ==> dimless
    #K_IPC ==> 3x3 kernel
    #a_BFE ==> 3x3 kernel
    #QE ==> Quantum Efficiency e-/photon
    #readout ==> readout noise std deviation    
    #hot_factor ==> avg hot pixel multiplier above average

    #Full Data Cube
    num_t_steps = int(t_int//dt)
    data_cube = np.zeros((num_t_steps,size[0],size[1]))
    field_charges = np.zeros(size)
    #field_intensity = np.zeros(size)
    
    #Add bad pixels and hot pixels
    if (np.size(hot_pixel_map) > 0 and np.size(bad_pixel_map) > 0 and (0 in bad_pixel_map or 1 in hot_pixel_map)):
        print("Adding Bad and Hot Pixels")
        field_charges = np.multiply(np.random.poisson(hot_factor*avg_intensity*dt*QE,size=size),hot_pixel_map)
        bad_hot_pixel_mask = bad_pixel_map*(1-hot_pixel_map)
        if (plot):
            plt.figure()
            plt.imshow(bad_hot_pixel_mask)
            plt.colorbar()
            plt.show()
    else:
        bad_hot_pixel_mask = np.ones(size)
    #Initialize with reset charge and bias frame
    if (bias_charge != 0):
        print("Adding Bias Charge " + str(bias_charge)) 
        field_charges += (bias_charge*np.ones(size))*bad_hot_pixel_mask #bias
    if (reset_charge != 0):
        print("Adding Reset Charge with Mean " + str(reset_charge))
        field_charges += np.random.poisson(reset_charge,size=size)
    
    data_cube[0,:,:] = copy.deepcopy(field_charges)
    
    #Create flat field of random intensity values using Gaussian random number generator; account for saturation; calculate gradient matrix with mean avg_intensity
    gradient = np.ones(size)
    
    if (len(intensity_gradient_x) > 0 or len(intensity_gradient_y) > 0):
        print("Adding Gradient in X direction")
        x_gradient = np.ones(size[1])
        for i in range(len(intensity_gradient_x)):
            coeff = intensity_gradient_x[i]
            if (i == 0):
                x_gradient *= coeff
            else:
                x_gradient += coeff*(np.power(np.arange(gradient_offset_x,size[1]+gradient_offset_x),i))
	    
        print("Adding Gradient in Y direction")
        y_gradient = np.ones(size[0])
        for j in range(len(intensity_gradient_y)):
            coeff = intensity_gradient_y[j]
            if (j == 0):
                y_gradient *= coeff
            else:
                y_gradient += coeff*(np.power(np.arange(gradient_offset_x,size[0]+gradient_offset_x),j))
    
        combined_grad = np.outer(y_gradient,x_gradient)
        gradient = combined_grad/np.mean(combined_grad)
        print(gradient)

    #rng = np.random.default_rng()
    #field_intensity = np.random.poisson(avg_intensity*gradient,size=size)
    #print(field_intensity)
    #print(np.max(field_intensity))
    #field_intensity = np.zeros(size)
    #field_intensity[field_intensity >= sat_intensity] = sat_intensity
    #field_intensity[field_intensity < 0] = 0
    """
    if (plot):
        f = plt.figure()
        plt.imshow(field_intensity)
        plt.colorbar()
        plt.title("Gaussian Flat Field Intensity")
    """
    
    #At each time step, sample charge collected (Poisson) and BFE
    field_charges += np.random.poisson(dt*avg_intensity*QE*gradient)*bad_hot_pixel_mask
    data_cube[1,:,:] = copy.deepcopy(field_charges)
    """
    #Add bad pixels and hot pixels
    field_charges = np.multiply(field_charges,bad_pixel_map)
    field_charges = field_charges - np.multiply(field_charges,hot_pixel_map) + np.multiply(np.random.poisson(hot_factor*avg_intensity*dt*QE),hot_pixel_map)#sat_charges*hot_pixel_map
    """
    #field_charges[field_charges >= sat_charges] = sat_charges
    #field_charges[field_charges < 0] = 0
    
    if (plot):
        f = plt.figure()
        plt.imshow(field_charges)
        plt.colorbar()
        plt.title("Bad and Hot Pixels added")
    
    for k in range(num_t_steps-2):
        #Convert Intensities to Charges with integration time steps
        
        #Add BFE effects ==> convolve a_BFE kernel with charges, add 1, convert to charge offsets
        #Method from Choi et al solid-waffle-master flat simulator
        if (np.sum(a_BFE != 0) > 0): 
            print("Adding BFE Step "+ str(k))
            W_BFE = 1+np.array(myConvolve(a_BFE,field_charges))
            field_charges += np.random.poisson(QE*avg_intensity*dt*W_BFE,size=size)*bad_hot_pixel_mask
        else:
            print("Adding Charge Timestep "+ str(k))
            field_charges += np.random.poisson(QE*avg_intensity*dt, size=size)*bad_hot_pixel_mask
        #Add shot noise proportional to sqrt(signal)
        #field_charges += np.random.normal(0,np.sqrt(field_charges),size=size)

        data_cube[k+2,:,:] = copy.deepcopy(field_charges)
       
         
        #field_charges += np.random.poisson(QE*avg_intensity*dt,size=size)*bad_hot_pixel_mask
        #data_cube[k+2,:,:] = copy.deepcopy(field_charges)
        """
        BFE_means = W_BFE*avg_intensity*dt#*field_charges
        BFE_offsets = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                BFE_offsets[i,j] = np.random.normal(BFE_means[i,j],np.sqrt(BFE_means[i,j]))
        field_charges += BFE_offsets*(bad_pixel_map + (1-hot_pixel_map))
        field_charges[field_charges >= sat_charges] = sat_charges
        field_charges[field_charges < 0] = 0
        """
    print(np.mean(field_charges))
    if (plot):
        f = plt.figure()
        plt.imshow(field_charges)
        plt.colorbar()
        plt.title("Charge accumulated and BFE")
    
    #Add IPC effects
    if (not (np.sum(K_IPC != 0) == 1 and K_IPC[K_IPC.shape[0]//2,K_IPC.shape[1]//2] == 1)):
        print("Adding IPC")
        for i in range(num_t_steps):
            data_cube[i,:,:] = myConvolve(K_IPC,data_cube[i,:,:])
        field_charges = copy.deepcopy(data_cube[-1,:,:])
    
        if (plot):
            f = plt.figure()
            plt.imshow(field_charges)
            plt.colorbar()
            plt.title("IPC Effects Added")

    #Add Nonlinear Term
    if (B != 0):
        print("Adding Nonlinearity B = " + str(B))
        data_cube -= (1e-6)*B*(np.multiply(data_cube,data_cube))
        field_charges = copy.deepcopy(data_cube[-1,:,:])#(1e-6)*B*(field_charges*field_charges)
        #field_charges[field_charges >= sat_charges] = sat_charges
        #field_charges[field_charges < 0] = 0
    
        if (plot):
            f = plt.figure()
            plt.imshow(field_charges)
            plt.colorbar()
            plt.title("Nonlinear Effects Added")
    
    #Convert to Signal ADU with gain and add readout noise
    """
    data_cube_signal = np.zeros(np.shape(data_cube))
    for i in range(num_t_steps):
        noise = np.random.normal(0,readout,size=size)
        data_cube_signal[i,:,:] += noise#np.random.normal(0,np.sqrt(data_cube_signal[i,:,:]),size=size)
    data_cube_signal = data_cube_signal/gain
    """
    if (readout != 0):
        print("Adding Readout Noise")
        data_cube = data_cube + np.random.normal(0,readout,size=np.shape(data_cube))
    
    data_cube_signal = data_cube/gain
    field_signal = copy.deepcopy(data_cube_signal[-1,:,:])#field_charges/gain 
    #field_signal += np.random.normal(0,readout)
    
    if (plot):
        f = plt.figure()
        plt.imshow(field_signal)
        plt.colorbar()
        plt.title("Converted to Signal")
    
    #Add saturation
    if (sat_charges >= 0):
        print("Adding Saturation at Signal Level " + str(sat_charges/gain))
        data_cube_signal = np.clip(data_cube_signal,0,sat_charges/gain)
        field_signal = copy.deepcopy(data_cube_signal[-1,:,:])

        if (plot):
            f = plt.figure()
            plt.imshow(field_signal)
            plt.colorbar()
            plt.title("Saturation Added")

    #Take cds frame
    cds_frame = field_signal - data_cube_signal[0,:,:]

    if (save_fits):
        saveToFits(data_cube_signal,(num_t_steps,size[0],size[1]),[gain,avg_intensity,a_BFE,K_IPC,B,QE,t_int],filename)
    
    return cds_frame# field_signal

#Adapted from solid-waffle 'calc_area_defect'
def myConvolve(kernel,image,npad=2):
    image_pad = np.pad(image,pad_width=(npad,npad),mode='symmetric')
    image_conv = signal.convolve(image_pad,kernel)#mode='same')
    extra_dim = (2*npad+kernel.shape[0]-1)//2
    return image_conv[extra_dim:-extra_dim,extra_dim:-extra_dim]


def saveToFits(data,size,arguments,name=""):
    filename = 'simulated_Flat_Frames/cmost_sim_flatframe_' + str(int(datetime.now().timestamp())) + name + '.fits'
    #create header
    header = fits.Header()
    header['FILENAME'] = filename
    header['EXP_TIME'] = (arguments[6]*1000, 'exposure time in msec')
    header['GAIN'] = (arguments[0], 'gain in e-/adu')
    header['INTENS'] = (arguments[1],'incident intensity in ph/s/pixel')
    header['BFE'] = (str(arguments[2][0,0]) +" " + str(arguments[2][1,0]) + " " + str(arguments[2][2,0]) + " " + str(arguments[2][0,1]) + " " + str(arguments[2][1,1]) + " " + str(arguments[2][2,1]) + " " + str(arguments[2][0,2]) + " " + str(arguments[2][1,2]) + " " + str(arguments[2][2,2]), 'BFE Kernel')
    header['IPC'] = (str(arguments[3][0,0]) +" " + str(arguments[3][1,0]) + " " + str(arguments[3][2,0]) + " " + str(arguments[3][0,1]) + " " + str(arguments[3][1,1]) + " " + str(arguments[3][2,1]) + " " + str(arguments[3][0,2]) + " " + str(arguments[3][1,2]) + " " + str(arguments[3][2,2]), 'IPC Kernel')
    header['NONLIN_B'] = (arguments[4],'Nonlinearity factor (quadratic)')
    header['QE']= (arguments[5],'Quantum Efficiency e-/ph')
    header['READOUTM'] = 'ROLLINGRESET'
    header['DATE'] = (datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),'FITS file write time')  
    hdu_prime = fits.PrimaryHDU(header=header)
    hdulist = [hdu_prime]
        
    for i in range(size[0]):
        hdu = fits.ImageHDU(data[i,:,:])
        hdulist.append(hdu)
            
    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(filename) 
