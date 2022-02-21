import nibabel as nib
import gzip
import numpy as np
import os
import shutil

class Reader():
    def __init__(self):
        pass

    def read_nii(self, path):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(path)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        return array

    def read_gzip(self, path):
        '''
        Reads .nii file from gzip and returns pixel array
        '''
        with gzip.open(path, 'rb') as f_in:
            with open(path[:-3]+'.nii', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        ct_scan = nib.load(path[:-3]+'.nii')
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        os.remove(path[:-3]+'.nii')
        return array

    def read(self, path):
        if path.endswith('nii'):
            return self.read_nii(path)
        if path.endswith('gz'):
            return self.read_gzip(path)
        return None
