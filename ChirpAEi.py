from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow import keras
import os, re, os.path
import skimage
import numpy
import librosa
import sys
import shutil
import numpy as np

def chirpaei_main(path):

    SOUND_DIR = '/Users/mm43533/INFERENCE/AcadianFlycatcher/input.mp3'
    INFERENCE_PATH = '/Users/mm43533/INFERENCE/AcadianFlycatcher/'
    shutil.copyfile(path,SOUND_DIR)
    
    ####### MP3 TO IMAGE ###################

    import os
    import skimage
    import numpy
    import librosa 

    # Plot mel-spectrogram
    N_FFT = 1024        
    HOP_SIZE = 1024      
    N_MELS = 128           
    WIN_SIZE = 1024     
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'     
    FMIN = 0


    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def spectrogram_image(mels, out):
        # use log-melspectrogram
        #mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
        mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)
        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
        img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy
        # save as PNG
        skimage.io.imsave(out, img)

    signal, sr = librosa.load(SOUND_DIR,duration=10) # sr = sampling rate
    S = librosa.feature.melspectrogram(y=signal,sr=sr,
                                n_fft=N_FFT,
                                hop_length=HOP_SIZE, 
                                n_mels=N_MELS, 
                                htk=True, 
                                fmin=FMIN, 
                                fmax=sr/2) 
    spectrogram_image(S,SOUND_DIR.replace('mp3','png'))

    ####### LOAD MODEL ###################

    model = keras.models.load_model('/Users/mm43533/MODELS/chirp_aei_model_35_class_0.7918')

    ####### CLEAN UP MP3 ############
    os.remove(SOUND_DIR)

    ####### PREPROCESS IMAGE ###################

    IM_SIZE = (224,224) 
    BIRDS = ['AcadianFlycatcher','AfricanBarredOwlet','AlderFlycatcher','AmericanBarnOwl','AmericanBittern','AmericanBlackDuck','AmericanCrow','AmericanGoldenPlover','AmericanHerringGull','AmericanKestrel','AmericanOystercatcher','AmericanWigeon','AmericanWoodcock','BarnSwallow','BarredOwl','BarredOwlet-nightjar','BeardedBellbird','BeltedKingfisher','Black-backedWoodpecker','Black-belliedWhistlingDuck','Black-billedCuckoo','Black-cappedChickadee','Black-crownedNightHeron','Black-neckedStilt','BlackRail','Blue-greyGnatcatcher','Blue-headedVireo','BlueJay','BohemianWaxwing','BorealChickadee','BrantGoose','Broad-wingedHawk','BrownCreeper','BrownThrasher','CanadaWarbler']
    DATA_PATH = '/Users/mm43533/INFERENCE/'
    BATCH_SIZE = 16


    from shutil import copy
    #copy(path,COPY_PATH)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_batches = test_datagen.flow_from_directory(DATA_PATH,
                                                      classes=BIRDS,
                                                      target_size=IM_SIZE,
                                                      class_mode='categorical', shuffle=False,
                                                      batch_size=BATCH_SIZE)

    ####### PREDICT ###################
    ypred = model.predict(test_batches)
    
    ####### POST PROCESS #############
    ypred1 = numpy.delete(ypred, ypred.argmax())
    ypred2 = numpy.delete(ypred1, ypred1.argmax())

    if ypred.max()*100> 45:
        print("\033 Top predictions based on your recording:\n")
        print(BIRDS[ypred.argmax()] + " with probability " + str(ypred.max()*100)+"%")
    else:
        print("No Bird Found")
    if ypred1.max()*100> 40:
        print(BIRDS[ypred1.argmax()] + " with probability " + str(ypred1.max()*100)+"%")
    if ypred2.max()*100> 30:
        print(BIRDS[ypred2.argmax()] + " with probability " + str(ypred2.max()*100)+"%")
        
    ####### CLEAN UP ############
    for root, dirs, files in os.walk(INFERENCE_PATH):
        for file in files:
            os.remove(os.path.join(root, file))
            
if __name__ == "__main__":
  recording_path = sys.argv[1]
  chirpaei_main(recording_path)
  print("Thank you for using ChirpAEI")

