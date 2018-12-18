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
    
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    if h5pyFile==None and memory_check==True: 
        if n<10000 and m<10000 and s<1000:
            cov = np.dot(center_matrix(x).T,center_matrix(y))
            return cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
        else:
            raise OverflowError ("Input matrices are probably too large")
    
    if datasetName==None:
        datasetName = "corrdata_%s" % datetime.datetime.today()
         
    cov = np.dot(center_matrix(x).T,center_matrix(y))
    r = cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])
    del cov

    h5pyFile.create_dataset(
            datasetName,
            shape=(x.shape[1], y.shape[1]),
            dtype="f",
#                compression="gzip",
            compression="lzf",
#                chunks=(100,y.shape[1]),
            data=r
            )
    del r
    return h5pyFile

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

if __name__ == "__main__":
# Testing performance
    import h5py
    import nilearn
    from nilearn.input_data import NiftiMasker as masker
    nilearn.EXPAND_PATH_WILDCARDS = False
    
    test_file = "test_scan.nii.gz"
    output_file = "test_subject_super_corr2.hdf5"
    
    f = h5py.File(output_file, "w")
    print("Creating hdf5 file: %s" % datetime.datetime.now())
    f.close()
    
    brain_masker = masker(verbose=0)
    cing_masker = masker("aal2_cingulate.nii.gz",verbose=0)
    
    brain_ts = brain_masker.fit_transform(test_file)
    cing_ts = cing_masker.fit_transform(test_file)
    print("Masking data: %s" % datetime.datetime.now())
    
    chunk_size = 3000
    chunks = chunk_getter(cing_ts.shape[1], chunk_size)
    
    for chunk in range(chunks):
        key = "cingulate_chunk_%03d" % chunk
        colrange = colrange_getter(cing_ts.shape[1], chunk, chunk_size)
        
        f = h5py.File(output_file, "r+")
        generate_correlation_map(cing_ts[:,colrange], brain_ts, f, key)
        f.close()
        print("Calc/save for chunk %d: %s" % (chunk, datetime.datetime.now()))
    
#    from multiprocessing import Pool  
#    chunks = chunk_getter(cing_ts.shape[1])
#    p = Pool()
#    p.map(generate_correlation_map(cing_ts[:,colrange_getter(cing_ts.shape[1],chunk)],brain_ts,f) for chunk in range(chunks))
