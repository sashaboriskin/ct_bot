import telebot
import config
import gzip
import nibabel as nib
import shutil
import os
#import segmentation
import matplotlib.pyplot as plt
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

TOKEN=config.TOKEN
bot = telebot.TeleBot(TOKEN)

def read_nii_gz(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    with gzip.open(filepath, 'rb') as f_in:
      with open(filepath[:-3]+'.nii', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    ct_scan = nib.load(filepath[:-3]+'.nii')
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    os.remove(filepath[:-3]+'.nii')
    os.remove(filepath)
    return array

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    os.remove(filepath)
    return array


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    #try:
    chat_id = message.chat.id

    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    arr = ''
    result = ''
    lung = ''
    ct = ''
    path = message.document.file_name
    src = 'C:/Users/sasha/PycharmProjects/Medbot/files' + path
    if path.endswith('.gz'):
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        arr = read_nii_gz(src)

    if path.endswith('.nii'):
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        arr = read_nii(src)
    bot.reply_to(message, "Ожидайте...")
    result, lung, ct = segmentation.segmentation(arr)
        
    
    print(result.shape)
    height = arr.shape[2]
    for i in range(height):
        fig = plt.figure(figsize = (24, 20))
        plt.subplot(1,3,1)
        plt.imshow(ct[i, ...,0], cmap = 'bone')
        plt.title('original lung')
        
        plt.subplot(1,3,2)
        plt.imshow(lung[i,...,0], cmap = 'bone')
        plt.title('lung')

        plt.subplot(1,3,3)
        plt.imshow(ct[i,:,:,0], cmap = 'bone')
        plt.imshow(result[i,:,:,0],alpha = 0.5,cmap = "nipy_spectral")
        plt.title('predicted infection mask')
        plt.savefig('./resfig/fig.png')
        bot.send_photo(message.chat.id, photo=open('./resfig/fig.png', 'rb'))
        os.remove('./resfig/fig.png')
#    except Exception as e:
 #       print(message, e)

@bot.message_handler(commands=['start', 'help'])
def start(message):
    bot.reply_to(message, "Пришлите .nii файл")

if __name__=="__main__":
    bot.polling(none_stop=True)
