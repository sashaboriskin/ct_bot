import numpy as np
import tensorflow as tf
import cv2

class Segmenter():
    def __init__(self):
        self.model_path = './unet_COVID_lung_model512x512_76_74_AUC98.h5'
        self.lung_model_path = './lung_unet_model512x512_0222_0217.h5'
        self.model = tf.keras.models.load_model(self.model_path)
        self.lung_model = tf.keras.models.load_model(self.lung_model_path)
        self.threshold = 0.01
    
    def preprocessing(self, ct_scan):
        arr = ct_scan
        slicearr = [cv2.resize(arr[:,:,ii], dsize=(512, 512), interpolation=cv2.INTER_AREA).astype('uint8')[..., np.newaxis] for ii in range(arr.shape[2])]
        preprocess_scan = np.asarray(slicearr)
        return preprocess_scan

    def segmentation(self, ct_scan):
        resarr = self.preprocessing(ct_scan)
        ct_slices = resarr.copy()
        resarr = np.array(resarr)/float(np.max(resarr))
        lung = self.lung_model.predict(resarr)
        mask = lung<0.01
        resarr[mask] = 0
        return self.model.predict(resarr), resarr, ct_slices 
