import numpy as np
import tensorflow as tf
import cv2

def segmentation(arr):
    model = tf.keras.models.load_model('C:/Users/sasha/PycharmProjects/Medbot/unet_COVID_lung_model512x512_76_74_AUC98.h5')
    lung_model = tf.keras.models.load_model('C:/Users/sasha/PycharmProjects/Medbot/lung_unet_model512x512_0222_0217.h5')
    print(arr.shape)
    resarr = np.asarray([cv2.resize(arr[:,:,ii], dsize=(512, 512), interpolation=cv2.INTER_AREA).astype('uint8')[..., np.newaxis] for ii in range(arr.shape[2])])
    ct_slices = resarr.copy()
    resarr = np.array(resarr)/float(np.max(resarr))
    lung = lung_model.predict(resarr)
#    print(*np.unique(resarr))
    mask = lung<0.01
    resarr[mask] = 0
    return model.predict(resarr), resarr, ct_slices 
