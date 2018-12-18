# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:28:58 2018

Updates:
    Batched to cluster
    Running batches from __main__
    Privatized paths
"""

import os
import h5py
import numpy as np
import nilearn
import datetime
import pandas as pd
from nilearn.input_data import NiftiMasker as masker
nilearn.EXPAND_PATH_WILDCARDS = False

def center_matrix(a):
    """
    Subtracts the column means from each column in a
    """
    mu_a = a.mean(0)    
    return np.subtract(a,np.reshape(np.repeat(mu_a,a.shape[0]),a.shape,order="F"))

def generate_correlation_map(x,y,h5pyFile=None,datasetName=None,small_matrices=False):
    """
    Function for correlating two matrices with uneven number of columns
    
    Parameters
    ----------
    x : np.array
      Shape T X N.

    y : np.array
      Shape T X M.
      
    Optional
    --------
    h5pyFile : h5py file || filetype:".hdf5"
    
    datasetName : string
      If no name is provided, the dataset will automatically be named
    
    small_matrices : bool
      Flag for memory operations
        True: override h5py requirement <-- thar be danger matey
        False: x and y are large matrices
        
      If you have large matrices, provide a hdf5 file
    """
    n = x.shape[1] #numcols of x
    m = y.shape[1] #numcols of y
    s = x.shape[0] #num subjects/timepoints
    
    if s != y.shape[0]:
        raise ValueError ("x and y must have the same number of observations")
    
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    if h5pyFile==None: 
        if n<12000 and m<12000 and s<1200:
            cov = np.dot(center_matrix(x).T,center_matrix(y))
            return cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
        elif small_matrices==True:
            cov = np.dot(center_matrix(x).T,center_matrix(y))
            return cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
        else:
            raise OverflowError("Inputs are probably too large, provide a hdf5 file")
            
    if datasetName==None:
        datasetName = "corrdata_%s" % datetime.datetime.today()
         
    cov = np.dot(center_matrix(x).T,center_matrix(y))
    r = cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
    del cov
    f = h5py.File(h5pyFile,"a")
    f.create_dataset(
            datasetName,
            shape=(x.shape[1], y.shape[1]),
            dtype="f",
            compression="lzf",
            data=r)
    f.close()
    del r
    
def chunk_getter(maxcol, chunk_size=1000):
    """
    Calculate number of chunks to divide x_cols into
    Default chunk_size is 1000 variables per chunk
    """
    chunks = 1
    while(maxcol/chunks) > chunk_size:
        chunks += 1
    return chunks

def colrange_getter(maxcol, chunk, chunk_size=1000):
    """
    Get the range of x_cols to grab
    """
    colrange = range(chunk*chunk_size, (chunk + 1)*chunk_size)
    if max(colrange) >= maxcol:
        colrange = range(chunk*chunk_size, maxcol)
    return colrange

def cingulate_voxelwise():
    """
    First-level voxel-voxel correlations for the cingulate cortex
    """  
    #Batch code
    with open("privatePaths.txt", 'r') as f:
        paths = [str(k).rstrip("\n") for k in f]
    subj_dir = r"%s" % paths[0]
    output_path = r"%s" % paths[1]
        
    chunk_dir =  r"%s" % paths[2]
    subj_batch = pd.read_csv("batch_1.csv",header=None,index_col=False)

    for s in subj_batch.values:
        s_str = str(s)
        subj = s_str.replace("[","").replace("'","").replace("]","")
        
        #Filename structure: denoised_######.nii
        subj_code = str(subj).split("_")[1].replace(".nii","")
        subj_nifti = os.path.join(subj_dir,subj)
        #print(subj_nifti)
        brain_masker = masker(verbose=1)
        brain_ts = brain_masker.fit_transform(subj_nifti)
        #print("Brain loaded")
        
        cing_files = os.listdir(chunk_dir)
        for cing_file in cing_files:    
            cing_mask = os.path.join(chunk_dir,cing_file)
            cing_masker = masker(cing_mask, verbose=1)
            cing_ts = cing_masker.fit_transform(subj_nifti)
            
            chunk_size = 200
            chunks = chunk_getter(cing_ts.shape[1], chunk_size)         
            
            cing_name = str(cing_file).replace(".nii.gz","")
            outfile = os.path.join(output_path, "%s_%s.hdf5" % (subj_code,cing_name))
            for chunk in range(chunks):
                colrange = colrange_getter(cing_ts.shape[1], chunk, chunk_size)
                chunk_name = "columns %d to %d" % (min(colrange),max(colrange))
                generate_correlation_map(cing_ts[:, colrange], brain_ts, outfile,chunk_name)
            del cing_ts  
            #break #performance testing
        
        del brain_ts
        #break #performance testing
        
if __name__ == "__main__":
    cingulate_voxelwise()