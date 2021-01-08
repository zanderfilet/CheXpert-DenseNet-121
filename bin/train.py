import pandas as pd
import numpy as np
import random

#CUDA -->>
#Parser for command-line arguments

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score


# from tensorflow.python.keras.utils.vis_utils import model_to_dot
# from IPython.display import SVG, display
# import pydot
# import graphviz

import pandas as pd
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "low_sample_test_model"

class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices']
input_shape = 320, 320, 1
n_classes = len(class_names)
n_epochs = 10


# Creating Densenet121
def densenet(input_shape, n_classes, filters=32):
    # batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x

    def dense_block(x, repetition):

        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x

    def transition_layer(x):

        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='sigmoid')(x)

    model = Model(input, output)
    return model

def data_setup():

    train_images = []
    train_classes = []

    data_raw = pd.read_csv("./config/train.csv")

    n_errors = 0
    errors = set()

    print("[DATA SETUP]")

    for index, row in tqdm(data_raw.iterrows()):
        im_path = "./DenseNet-im/{}".format(row["Path"])

        try:
            im = Image.open(im_path)
            matrix = np.array(im.getdata())
            normalized_matrix = np.true_divide(matrix, 255)
            zero_mean_mtrx = normalized_matrix - np.mean(normalized_matrix)
            standardized_matrix = zero_mean_mtrx / np.std(normalized_matrix)

            classes_v = [row['Atelectasis'], row['Cardiomegaly'], row['Consolidation'], row['Edema'], row['Enlarged Cardiomediastinum'],
                           row['Fracture'], row['Lung Lesion'], row['Lung Opacity'], row['No Finding'], row['Pleural Effusion'], row['Pleural Other'],
                           row['Pneumonia'], row['Pneumothorax'], row['Support Devices']]

            train_images.append(standardized_matrix)
            train_classes.append(classes_v)

        except:
            n_errors += 1
            errors.add(row["Path"])

    print("Done. Encountered {} errors. Errors:".format(n_errors))
    # print(errors)

    return train_images, train_classes

def rm_neg_par(train_images, train_classes):
    # If Par. = 0, Child != 0 -> REMOVE

    #### 0  'Atelectasis',
    #### 1  'Cardiomegaly',
    #### 2  'Consolidation',
    #### 3  'Edema',
    #### 4  'Enlarged Cardiomediastinum',
    #### 5  'Fracture',
    #### 6  'Lung Lesion',
    #### 7  'Lung Opacity',
    #### 8  'No Finding',
    #### 9  'Pleural Effusion',
    #### 10 'Pleural Other',
    #### 11 'Pneumonia',
    #### 12 'Pneumothorax',
    #### 13 'Support Devices'

    # Enlarged Cardiomegaly  -->  Cardiomegaly  [1] [4]
    # Lung Opacity           -->  Edema         [7] [3]
    # Lung Opacity           -->  Consolidation [7] [2]
    # Lung Opacity           -->  Pneumonia     [7] [11]
    # Lung Opacity           -->  Lesion        [7] [6]
    # Lung Opacity           -->  Atelectasis   [7] [0]
    # Consolidation          --> Pneumonia      [2] [11]

    print("[NEGATIVE PARENTS FILTER]")

    n_removed_labels_im = 0
    removed_labels_im = []
    removed_labels_cl = []

    labels_im = []
    labels_cl = []

    # [ PARENT, CHILD ]
    negative_parents = [[1, 4], [7, 3], [7, 2], [7, 11], [7, 6], [7, 0], [2, 11]]
    filtered = 0

    for i in tqdm(range(len(train_classes))):
        n_par = False
        for combination in negative_parents:
            if train_classes[i][combination[1]] != 0.0 and train_classes[i][combination[0]] == 0.0:
                filtered += 1
                n_par = True
                break

        if not n_par:
            removed_labels_cl.append(train_classes[i])
            removed_labels_im.extend(train_images[i])
            n_removed_labels_im += 1

        labels_cl.append(train_classes[i])
        labels_im.extend(train_images[i])

    print("Done. {} / {} filtered.".format(filtered, len(train_classes)))

    return np.asarray(removed_labels_im).reshape(n_removed_labels_im,320,320,1), np.asarray(removed_labels_cl), np.asarray(labels_im).reshape(len(train_classes),320,320,1), np.asarray(train_classes)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def model_train(train_images, train_classes):
    model = densenet(input_shape, n_classes)

    # model.summary()

    # Initially 0.0001
    lr = 0.001

    optimizer = optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=['accuracy', auroc])

    t_im_no_neg_par, t_cl_no_neg_par, full_train_images, full_train_classes = rm_neg_par(train_images, train_classes)

    # print(full_train_images)

    for i in range(n_epochs):
        if i < 5:
            model.fit(t_im_no_neg_par, t_cl_no_neg_par, initial_epoch=i, epochs=i+1, verbose=1, batch_size=32)
        else:
            model.fit(full_train_images, full_train_classes, initial_epoch=i, epochs=i+1, verbose=1, batch_size=32)

        model.save('./models/{}-e{}'.format(MODEL_NAME, i))

        lr *= 0.1
        K.set_value(model.optimizer.learning_rate, lr)

    # model.summary()
    # display(SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='TB',expand_nested=False, dpi=60, subgraph=False).create(prog='dot', format='svg')))

    return model

def main():
    train_images, train_classes = data_setup()
    model_train(train_images, train_classes)

if __name__ == "__main__":
    main()

 