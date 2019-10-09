'''
Ideas for paper
1) Random noise generation
2) Empirical noise generator
    a) generated by tisse type
    b) across the entire dataset
    c) Local noise generator

Ideas:
RANDOM NOISE GENERATION


'''

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from model import unet

def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 20.0
    image[:,:,1] -= 85.1
    image[:,:,2] -= 94.17
    return image

def return_channels(array):
    """
    Helper function to return channels
    """
    return array[:,:,0], array[:,:,1], array[:,:,2]

def min_max_rescaling(array, percentile_clip = 3):
    p_low, p_high = np.percentile(array, (percentile_clip, 100 - percentile_clip))
    array = array.clip(min = p_low, max = p_high)
    img = (array - p_low)/(p_high - p_low)
    return img

def channel_rescaling(array):
    DNA, CH2, CH3 = return_channels(array.astype('float'))
    img = np.empty((array.shape[0], array.shape[1], 3), dtype='float')

    img[:,:,0] = min_max_rescaling(DNA)
    img[:,:,1] = min_max_rescaling(CH2)
    img[:,:,2] = min_max_rescaling(CH3)

    img *= 255
    
    return img.astype(np.uint8)

def rescale(img):
    rescaled_image = (img - img.min())/(img.max() - img.min()) * 255
    return rescaled_image.astype(np.uint8)

def uniform_noise_generator(batch, sigma = 100):
    # return image parameters
    batch_size, height, width, channels = batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]

    # generate random uniform noise
    noise = np.random.random_sample(size=(batch_size, height, width, channels))

    batch = batch + (noise * sigma)   
    # batch = batch.clip(min=-127, max=127)    


    return batch

def gaussian_noise_generator(batch, sigma_range = (25,100)):
    # return image parameters
    batch_size, height, width, channels = batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]

    # generate random uniform noise
    sigma = np.random.randint(low = sigma_range[0], high = sigma_range[1])
    noise = np.random.normal(loc=0.0, scale=sigma, size=(batch_size, height, width, channels))
    # batch = batch.clip(min=-127, max=127)    

    return batch + noise

def denoising_generator(generator, noise_function = gaussian_noise_generator):
    for batch in generator:
        noisy_batch = noise_function(batch)
        yield noisy_batch, batch


input_img = Input(shape=(HEIGHT, WIDTH, CHANNELS))
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)
denoiser = Model(input_img, decoded)
denoiser.compile(optimizer = Adam(lr = 0.001), loss = 'mean_absolute_error', metrics = ['mae'])



if __name__ == "__main__":
    
    training_directory = "/home/todd/Desktop/SRH_genetics/srh_patches/patches/training_patches/training"

    HEIGHT, WIDTH, CHANNELS = 300, 300, 3
    BATCH_SIZE = 32

    # Define the generator
    train_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function = nio_preprocessing_function,
        data_format = "channels_last").flow_from_directory(directory = training_directory, 
        target_size = (HEIGHT, WIDTH), interpolation = "bicubic", color_mode = 'rgb', classes = None, class_mode = None, 
        batch_size = BATCH_SIZE, shuffle = True)
        # save_to_dir = "/home/todd/Desktop/test_dir/keras_save_dir")

    # Unet = unet(input_size = (HEIGHT, WIDTH, 3))
    
    adam = Adam(lr=0.0005)
    denoiser.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])

    denoiser.fit_generator(denoising_generator(train_generator),
                    epochs=10,
                    steps_per_epoch=600,
                    shuffle=True)


img_stack = next(train_generator)
noisy_img_stack = gaussian_noise_generator(img_stack)
decod_img_stack = denoiser.predict(noisy_img_stack)

index = 4
img = channel_rescaling(img_stack[index,:,:,:])
noisy_img = channel_rescaling(noisy_img_stack[index,:,:,:])
decod_img = channel_rescaling(decod_img_stack[index,:,:,:])

plt.imshow(np.hstack((img, noisy_img, decod_img)))
plt.show()

