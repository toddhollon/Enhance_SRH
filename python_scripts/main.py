

from model import *
from data import *
import numpy as np
import os
# from skimage.io import imread
from imageio import imread
import skimage.transform as trans
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras import backend as K
from scipy.misc import toimage
from pandas import DataFrame
from pylab import rcParams  

image_size = (256, 256, 3)

def preprocessing_rescale(img):
    '''
    Simple rescaling function to make pixel values between 0-1
    '''
    if (np.max(img) > 1):
        return img / 255    
    else:
        return img
    return img

def iou(y_true, y_pred):
    # must use keras tensorflow backend to compute these values because the output is a tf tensor

    # calculates a boolean array, then converts to float
    y_true = K.cast(K.greater_equal(y_true, 0.5), K.floatx())
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())

    # computes intersection and union
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # error handling if union is equal to 0, avoid zerodivision error
    return K.switch(K.equal(union, 0), 1.0, intersection/union)

def iou_metric(y_true_path, y_pred_path):
    y_true = imread(y_true_path)
    y_pred = imread(y_pred_path)

    y_true = trans.resize(y_true, (256, 256))[:,:,0]
    y_pred = trans.resize(y_pred, (256, 256))

    # computes intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    # error handling if union is equal to 0, avoid zerodivision error
    print(intersection/union)

def mask_path_modifiers(path):
    path = path.replace("images", "masks")
    path = path.replace("image", "mask")
    return path

def forward_pass(image_path, model):
    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0]
    return pred

def save_predictions(pred, file):
    binary = np.copy(pred)
    
    # save predictions as both greyscale and viridis colormap
    pred *= 255
    toimage(pred, cmin=0, cmax=255).save(file + "_pred.png")
    plt.imsave(file + "_viridis_pred.png", pred, cmap = "viridis", vmin = 0, vmax = 255)

    # save predictions as binary masks
    binary[binary > 0.90] = 1
    binary[binary <= 0.90] = 0
    binary *= 255
    toimage(binary, cmin=0.0, cmax=255).save(file + "_binary.png")

def batch_iou(preds, y_test):
    iou_val = []
    for i in range(len(preds)):
        def indexer(array):
            indices = []
            for index, val in enumerate(array):
                if val:
                    indices.append(index)
            return set(indices)

        pred_flat = indexer(preds[i,:,:,:].flatten())
        ground_flat = indexer(y_test[i,:,:,:].flatten())
        iou = len(pred_flat & ground_flat)/len(pred_flat | ground_flat)
        iou_val.append(iou)
    return iou_val

def plotting_function(image_path, model):

    img = imread(image_path)
    img_resize = trans.resize(img,(256, 256))
    img_for_net = np.expand_dims(img_resize, axis = 0)
    pred = model.predict(preprocessing_rescale(img_for_net))[0,:,:,0]

    fig = plt.figure(figsize=(20, 6), dpi=100)
    ax1 = fig.add_subplot(1,4,1)
    diff = 1 - img_resize.max()
    ax1.imshow(img_resize + diff)
    ax1.set_title("Oncostream image")

    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(pred, cmap='viridis')
    ax2.set_title("Prediction heatmap")

    ax3 = fig.add_subplot(1,4,3)
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    ax3.imshow(pred, cmap="Greys_r")
    ax3.set_title("Binarized prediction")

    mask = imread(mask_path_modifiers(image_path))
    mask = trans.resize(mask,(256, 256))
    ax4 = fig.add_subplot(1,4,4)
    ax4.imshow(mask, cmap="Greys_r")
    ax4.set_title("Ground truth")

    def indexer(array):
        indices = []
        for index, val in enumerate(array):
            if val:
                indices.append(index)
        return set(indices)

    pred_flat = indexer(pred.flatten())
    ground_flat = indexer(mask[:,:,0].flatten())
    iou = len(pred_flat & ground_flat)/len(pred_flat | ground_flat)
    plt.suptitle("Intersection over union: " + str(np.round(iou, 3)))
    plt.show()

def plotting_function_save(image_path, model):

    rcParams['figure.figsize'] = 17, 5
    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    diff = 1 - img_resize.max()
    ax1.imshow(img_resize + diff, vmin = 0, vmax = 1)
    ax1.set_title("Oncostream image")

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(pred, cmap='viridis', vmin = 0, vmax = 1)
    ax2.set_title("Prediction heatmap")

    ax3 = fig.add_subplot(1,3,3)
    pred[pred > 0.99] = 1
    pred[pred <= 0.99] = 0
    ax3.imshow(pred, cmap="Greys_r", vmin = 0, vmax = 1)
    ax3.set_title("Binarized prediction")

    plt.suptitle(image_path.split("/")[-1] + " Area oncosctream: " + str(np.round(calculate_area_oncostream(image_path, model), decimals = 3)))
    plt.savefig(image_path.split("/")[-1][0:-4] + ".png", dpi = (500))

def plotting_function_inference(image_path, model):

    rcParams['figure.figsize'] = 17, 5
    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    diff = 1 - img_resize.max()
    ax1.imshow(img_resize + diff, vmin = 0, vmax = 1)
    ax1.set_title("Oncostream image")

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(pred, cmap='viridis', vmin = 0, vmax = 1)
    ax2.set_title("Prediction heatmap")

    ax3 = fig.add_subplot(1,3,3)
    pred[pred > 0.99] = 1
    pred[pred <= 0.99] = 0
    ax3.imshow(pred, cmap="Greys_r", vmin = 0, vmax = 1)
    ax3.set_title("Binarized prediction")

    plt.suptitle(image_path.split("/")[-1] + " - Area oncosctream: " + str(np.round(calculate_area_oncostream(image_path, model), decimals = 3)))
    plt.show()

def calculate_area_oncostream(image_path, model):

    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!

    pred[pred > 0.99] = 1
    pred[pred <= 0.99] = 0
    onco_area = pred.sum()
    return onco_area/(256*256)

def export_oncostream_area(image_dir, model):
    filelist = sorted(os.listdir(image_dir))
    arealist = []
    for file in filelist:
        arealist.append(calculate_area_oncostream(os.path.join(image_dir, file), model))
    return DataFrame({"files":filelist, "areas":arealist})
    
def contour_plot(image_path, model):
    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!

    percentiles = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    pred = np.flipud(pred) # realign the y axis

    fig, ax = plt.subplots()
    CS = ax.contour(pred, levels = percentiles)
    ax.clabel(CS, inline = 1)
    plt.show()

def step_decay(epoch):
    initial_lr = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lr * np.power(drop, np.floor((epoch + 1)/epochs_drop))
    return lrate

def train_histories(epochs, validation_dir):
    """
    Returns a dataframe of the training history
    """
    model = load_model("/home/orringer-lab/Desktop/oncostreams/models/oncostreams_fcnn_97trainacc.hdf5")
    validation_generator = trainGenerator(batch_size = 4,
                        train_path = validation_dir,
                        image_folder = 'images', 
                        mask_folder = 'masks', 
                        aug_dict = data_gen_args, 
                        save_to_dir = None)

    lrate_schedule = LearningRateScheduler(step_decay)

    model.compile(optimizer = Adam(lr = learn_rate), loss = 'binary_crossentropy', metrics = ['accuracy', iou])
    history = model.fit_generator(train_generator, steps_per_epoch=300, epochs=epochs, validation_data=validation_generator, callbacks=[lrate_schedule], validation_steps=25, shuffle = True)

    return (DataFrame(history.history), model)

data_gen_args = dict(rotation_range = 0.0,
                    width_shift_range = 0.0,
                    height_shift_range = 0.0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode=None)

if __name__ == "__main__":

    model = load_model("/Volumes/UNTITLED/oncostreams/epochs_500/model_tranined_set_1.hdf5", custom_objects = {'iou':iou})
    plotting_function_inference("/Volumes/UNTITLED/TODD-2019-05-19/Image_81.tif", model)    

#     training_image_dir = '/home/orringer-lab/Desktop/oncostreams/random_crops'
#     validation_image_dir = "/home/orringer-lab/Desktop/oncostreams/validation_set"
#     validation_dir_list = sorted(os.listdir(validation_image_dir))
#     validation_dir_list = [os.path.join(validation_image_dir, x) for x in validation_dir_list]
# #    cell_dir = "/home/orringer-lab/Desktop/unet/tiled_images/train"
#     learn_rate = 0.00001
    
#     train_generator = trainGenerator(batch_size = 5,
#                             train_path = training_image_dir,
#                             image_folder = 'images', 
#                             mask_folder = 'masks', 
#                             aug_dict = data_gen_args, 
#                             save_to_dir = None)
    
#     for val_dir in validation_dir_list:
#         train_df, model = train_histories(epochs = 25, validation_dir = val_dir)
#         train_df.to_excel(val_dir.split("/")[-1] + ".xlsx")
#         model.save('model_trained_' + val_dir + ".hdf5")

    # df = export_oncostream_area("/Volumes/UNTITLED/TODD-2019-05-19", model)
    # df.to_excel("area.xlsx")


    image_dir = "/Volumes/UNTITLED/TODD-2019-05-19/"
    filelist = os.listdir(image_dir)
    for file in filelist:
        plotting_function_inference(os.path.join(image_dir, file), model)


