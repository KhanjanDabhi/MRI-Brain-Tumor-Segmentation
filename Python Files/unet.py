#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, load_model
from keras.layers import Input ,BatchNormalization , Activation 
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers 
from sklearn.model_selection import train_test_split
import os
import nibabel as nib
from keras import backend as K
import glob
import imageio
import skimage.io as io
import skimage.color as color
import random as r
import math
from nilearn import plotting
from PIL import Image
import pickle
import skimage.transform as skTrans
from nilearn import image
from nilearn.image import resample_img
import nibabel.processing
import warnings

def Convolution(input_tensor,filters):

    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x

def model(input_shape):
    
    inputs = Input((input_shape))
    
    conv_1 = Convolution(inputs,32)
    maxp_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_1)
    
    conv_2 = Convolution(maxp_1,64)
    maxp_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_2)
    
    conv_3 = Convolution(maxp_2,128)
    maxp_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_3)
    
    conv_4 = Convolution(maxp_3,256)
    maxp_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_4)
    
    conv_5 = Convolution(maxp_4,512)
    upsample_6 = UpSampling2D((2, 2)) (conv_5)
    
    conv_6 = Convolution(upsample_6,256)
    upsample_7 = UpSampling2D((2, 2)) (conv_6)
    
    upsample_7 = concatenate([upsample_7, conv_3])
    
    conv_7 = Convolution(upsample_7,128)
    upsample_8 = UpSampling2D((2, 2)) (conv_7)
    
    conv_8 = Convolution(upsample_8,64)
    upsample_9 = UpSampling2D((2, 2)) (conv_8)
    
    upsample_9 = concatenate([upsample_9, conv_1])
    
    conv_9 = Convolution(upsample_9,32)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_9)
    
    model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model

# Loding the Light weighted CNN
model = model(input_shape = (128,128,1))

model.load_weights(os.path.dirname(__file__) + '/base_model.h5')

"""TumorPredict(4- array for paths of files,1-location to store the output ) function"""
def rescale_Nii(nifti_file):
    warnings.filterwarnings("ignore")
    img=nifti_file
    #voxel_dims=[3.8, 3.8,1]
    voxel_dims=[1.60, 1.60,1]
    

    # downl sample to 128*128*155
    #target_shape=(64,64,130)
    target_shape=(128,128,130)
    # Initialize target_affine
    target_affine = img.affine.copy()
    # Calculate the translation part of the affine
    spatial_dimensions = (img.header['dim'] * img.header['pixdim'])[1:4]
    
    # Calculate the translation affine as a proportion of the real world
    # spatial dimensions
    image_center_as_prop = img.affine[0:3,3] / spatial_dimensions
    
    # Calculate the equivalent center coordinates in the target image
    dimensions_of_target_image = (np.array(voxel_dims) * np.array(target_shape))
    target_center_coords =  dimensions_of_target_image * image_center_as_prop
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)
    
    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims
    
    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    target_affine[:3,3] = target_center_coords 
  

    #target_affine = rescale_affine(target_affine,voxel_dims,target_center_coords)
    resampled_img = resample_img(img, target_affine=target_affine,target_shape=target_shape)
    resampled_img.header.set_zooms((np.absolute(voxel_dims)))
    return resampled_img


import sys
 
# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)
 
# Arguments passed
print("\nName of Python script:", sys.argv[0])

t1= sys.argv[1]
t1ce= sys.argv[2]
flair= sys.argv[3]
t2= sys.argv[4]



modalities=[t1,t2,flair,t2]

all_modalities = []    
for modality in modalities:      
        nifti_file   = nib.load(modality)
        nifti_file= rescale_Nii(nifti_file)
        brain_numpy  = np.asarray(nifti_file.dataobj)
        all_modalities.append(brain_numpy)
brain_affine   = nifti_file.affine
all_modalities = np.array(all_modalities)
all_modalities = np.rint(all_modalities).astype(np.int16)
all_modalities = np.transpose(all_modalities)
avg_modality=[]
for i in range(len(all_modalities)):
    x=(all_modalities[i,:,:,0]+all_modalities[i,:,:,1]+all_modalities[i,:,:,2]+all_modalities[i,:,:,3])/4
    avg_modality.append(x)  
avg_modality=np.array(avg_modality)







TR=np.array(avg_modality[:,:,:],dtype='float32')

pref_Tumor = model.predict(TR)


"""SaveOutput(location to save,pref_tumor) function."""
try:
    os.makedirs(sys.argv[5])

except:
    print("Dir already exsisted")

oldFiles = glob.glob(sys.argv[5] + "/*")

for file in oldFiles:
    os.remove(file)


# This part save the pred_Tumor array into the given file directory
for i in range(len(pref_Tumor)):
    fig = plt.figure(figsize=(5,5))
    plt.title('Slice Number:'+str(i+1))
    plt.imshow(np.squeeze(TR[i,:,:]),cmap='gray')
    plt.imshow(np.squeeze(pref_Tumor[i,:,:,0]),alpha=0.3,cmap='Reds')
    
    plt.savefig(sys.argv[5] + "/" + "slice_" + str(i+1).zfill(3).rjust(4, '0') + ".png")
    
  

#Output the .nii file
x=pref_Tumor[:,:,:,0]
x = np.transpose(x, (1,2,0))
img = nib.Nifti1Image(x, brain_affine)
nib.save(img, sys.argv[5] + "/Segmentation.nii.gz") 