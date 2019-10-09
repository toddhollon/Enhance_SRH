
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt

#from keras.datasets import mnist
#import numpy as np
#
#(x_train, _), (x_test, _) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


######################################
img_rows = 104
img_cols = 104
img_channels = 3

def autoencoder_preprocessing(array):
    array = array.astype('float32') / 255.
    return array

def autoencoder_generator(generator):
    for batch in generator:
        yield batch, batch

def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

# Convolutional autoencoder for images
train_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function = nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = "/home/todd/Desktop/CNN_Images/nio_validation_tiles/glioblastoma", 
    target_size = (img_rows, img_cols), interpolation = "bicubic", color_mode = 'rgb', classes = None, class_mode = None, 
    batch_size = 64, shuffle = True)
#    save_to_dir = "/home/orringer-lab/Desktop/keras_save_dir")


input_img = Input(shape=(img_rows, img_cols, img_channels))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit_generator(autoencoder_generator(train_generator),
                epochs=10,
                steps_per_epoch=600,
                shuffle=True)

encoder = Model(input_img, encoded)
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



img = next(train_generator)
decod_img = autoencoder.predict(img)
img = (img[0]*255).astype('uint8')
decod_img = (decod_img[0]*255).astype('uint8')
plt.imshow(np.hstack((img, decod_img)))



'''
We can also have a look at the 128-dimensional encoded representations. 
These representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images.
'''
# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(n):
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()