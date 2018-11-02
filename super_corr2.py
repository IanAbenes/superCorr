# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:08:52 2018

Function for voxelwise connectivity analysis

@author: ixa080020
"""
import numpy as np
import datetime

def center_matrix(a):
    """
    Subtracts the column means from each column in a
    """
    mu_a = a.mean(0)    
    return np.subtract(a,np.reshape(np.repeat(mu_a,a.shape[0]),a.shape,order="F"))

def generate_correlation_map(x,y,h5pyFile=None,datasetName=None,memory_check=True):
    """Correlate each column of X with each column of Y.

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
    
    memory_check : bool
      Flag for memory operations
        True: assumes that x and y are large matrices
        False: x and y are relatively small matrices
        
      If you have large matrices (large N and/or M and/or T), it is recommended
      that you provide a path to a hdf5 file
    
    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    Formula
    -------
    First, center x and y. Then 
      cov = np.dot(x, y.T)
      
      r = cov / np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])  
    """
    n = x.shape[1]
    m = y.shape[1]
    s = x.shape[0]
    
    if s != y.shape[0]:
        raise ValueError ("x and y must have the same number of observations")
    
    if h5pyFile==None and memory_check==True: 
        if n<10000 and m<100000 and s<1000:
            memory_check=False
        else:
            raise OverflowError ("Input matrices are probably too large")
    
    if datasetName==None:
        datasetName = "corrdata_%s" % datetime.datetime.today()
        
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    cov = np.dot(center_matrix(x).T,center_matrix(y))
    r = cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
    del cov
    
    if memory_check==False:
        return r
    else:
        h5pyFile.create_dataset(
                datasetName,
                shape=(x.shape[1], y.shape[1]),
                dtype="f",
#                compression="gzip",
                compression="lzf",
                chunks=(100,y.shape[1]),
                data=r
                )

        del r
        return h5pyFile

if __name__ == "__main__":
# Testing performance
    import os
    import h5py
    import nilearn
    from nilearn.input_data import NiftiMasker as masker
    nilearn.EXPAND_PATH_WILDCARDS = False
     
    submask_dir = os.path.join(".","cing_chunk12")
    submasks = sorted([f for f in os.listdir(submask_dir)])
    
    test_file = "test_scan.nii.gz"
    output_file = "test_subject_super_corr2.hdf5"
    
    brain_masker = masker(verbose=0)
    
    f = h5py.File(output_file, "w")
    print("Creating hdf5 file: %s" % datetime.datetime.now())
    f.close()
    
    brain_ts = brain_masker.fit_transform(test_file)
    print("Masking brain: %s" % datetime.datetime.now())
    
    for submask in submasks:
        f = h5py.File(output_file,"r+")
        
        key = "cingulate_chunk_%s" % submask.split("_")[0]
        cing_submask = masker(os.path.join(submask_dir,submask))
        
        submask_ts = cing_submask.fit_transform(test_file)
        print("Masking cingulate - %s: %s" % (key,datetime.datetime.now()))
        
        generate_correlation_map(submask_ts,brain_ts,f,key)
        
        del submask_ts
        print("Corr calculation + saving: %s" % datetime.datetime.now())
        f.close()