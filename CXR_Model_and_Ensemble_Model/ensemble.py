from keras.losses import MSE
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, SeparableConv2D
from keras.layers import Input, Add, Flatten, ReLU, Concatenate,ELU
from keras.models import Model
from keras.optimizers import RMSprop
from tqdm import tqdm
import numpy as np
import os
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
import tensorflow as tf
import logging
from keras.models import load_model
from keras.layers import Lambda
from keras.models import load_model
import pickle
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
import json

# Set TensorFlow logging level to suppress warnings
#tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")
from scipy.ndimage import zoom
def resize_with_ratio(image, dims, interpolationFlag):
    height, width = image.shape
    scale_factor = min(dims[0] / width, dims[1] / height)

    target_original_height = int(height * scale_factor)
    target_original_width = int(width * scale_factor)

    if interpolationFlag == "Spline":
        downscaled_image = zoom(image, scale_factor, order = 3)
    else:
        downscaled_image = cv2.resize(image, (target_original_width, target_original_height),
                                      interpolation=interpolationFlag)

    pad_top = (dims[1] - target_original_height) // 2
    pad_bottom = dims[1] - target_original_height - pad_top
    pad_left = (dims[0] - target_original_width) // 2
    pad_right = dims[0] - target_original_width - pad_left

    padded_image = cv2.copyMakeBorder(downscaled_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return padded_image

#clahe = cv2.createCLAHE(clipLimit = 2.5, tileGridSize = (4,4))

# Step 1: Load and preprocess the data
def load_data(directory, image_size=(250, 250)):
    images = []
    labels = []
    for label, category in enumerate(["NORMAL", "PNEUMONIA"]):
        category_path = os.path.join(directory, category)
        for file_name in tqdm(os.listdir(category_path)):
            image_path = os.path.join(category_path, file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_with_ratio(image, (600, 600), cv2.INTER_AREA)
            #image = image / 255.0
            #sigma = estimate_sigma(image, multichannel=False, average_sigmas=True)
            #image = denoise_nl_means(image, patch_size=4,
            #                         patch_distance=2,
            #                         h=0.2 * sigma, fast_mode=True)
            #image = (image*255).astype("uint8")
            #image = clahe.apply(image)
            #image = cv2.resize(image, image_size, interpolation = cv2.INTER_AREA)
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = image / 255.0
            image = np.expand_dims(image, axis=-1)
            images.append(image)
            if category == "PNEUMONIA":
                labels.append(1)  # Assign label 1 for PNEUMONIA images
            else:
                labels.append(0)  # Assign label 0 for NORMAL images
    return np.array(images), np.array(labels)




class MaxAccuracy(Callback):
    def __init__(self):
        super(MaxAccuracy, self).__init__()
        self.max_train_accuracy = 0.0
        self.max_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        train_accuracy = logs['accuracy']
        val_accuracy = logs['val_accuracy']
        if train_accuracy > self.max_train_accuracy:
            self.max_train_accuracy = train_accuracy
        if val_accuracy > self.max_val_accuracy:
            self.max_val_accuracy = val_accuracy
        print(
            f" Max Train Accuracy: {self.max_train_accuracy:.4f}, Max Validation Accuracy: {self.max_val_accuracy:.4f}")

def train_model(model, model_loc, x, y, n_splits=1, test_size=0.25, random_state=47, n_epochs=15,
                min_learning_rate=0.0000001, lr_decay_factor=0.8):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    aug_set_id = 0
    histories = []
    max_accuracy = MaxAccuracy()
    for train_ids, test_ids in splitter.split(x, y):
        checkpoint = ModelCheckpoint("model_" + str(model_loc) + "_" + str(aug_set_id) + ".h5", monitor="val_accuracy",
                                     save_best_only=True, model="max")
        aug_set_id = aug_set_id + 1

        x_train, x_test, y_train, y_test = x[train_ids], x[test_ids], y[train_ids], y[test_ids]

        histories.append(model.fit(x_train, y_train, batch_size=32, 
                                   epochs=n_epochs, validation_data=(x_test, y_test),
                                   callbacks=[ReduceLROnPlateau(monitor="val_accuracy", factor=lr_decay_factor,
                                                                patience=2, min_lr=min_learning_rate), checkpoint, max_accuracy]))

    return histories

images, labels = load_data(".", (600, 600))

labels = to_categorical(labels)

tf.config.run_functions_eagerly(True)

sub_models = []
sub_model_outputs = []
n_splits = 8
res = 600
factor = int(res/n_splits)
main_input = Input((600, 600, 1))
def extract_patch(input_image, n_model, size=factor):
    return input_image[:, int(n_model/n_splits)*factor:min((int(n_model/n_splits)+1)*factor, res), int(n_model%n_splits)*factor:min((int(n_model%n_splits)+1)*factor, res), :]

with open("best_model.json", "r") as file:
    model_list = json.load(file)

for n_model in model_list:
    model = load_model("model_"+str(n_model)+"_0.h5")
    for layer in model.layers[:-1]:
        layer.trainable = False
    new_output_layer = Dense(1, activation="sigmoid")(model.layers[-2].output)
    sub_model = Model(inputs = model.input, outputs = new_output_layer)
    start_row, start_col = (n_model // n_splits) * 60, (n_model % n_splits) * 60  # Adjust based on your actual layout
    patch = Lambda(extract_patch, arguments={'n_model': n_model})(main_input)

    sub_models.append(sub_model)
    sub_model_outputs.append(sub_model(patch))
concatenated_outputs = Concatenate()(sub_model_outputs)
dense_layer = Dense(64, activation='relu')(concatenated_outputs)
final_output = Dense(2, activation='softmax')(dense_layer)
ensemble_model = Model(inputs=main_input, outputs=final_output)
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

histories = train_model(ensemble_model, "universal", images, labels)


with open('histories_ensemble.pkl', 'wb') as f:
    pickle.dump(histories, f)