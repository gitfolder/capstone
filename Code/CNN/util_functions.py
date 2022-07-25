from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset
from keras.callbacks import History
from tensorflow.keras import activations
import tensorflow as tf
from tensorflow.keras import layers

from keras import backend as K
#from keras import callbacks

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical 

from io import StringIO
#from tqdm import tqdm



decay_rate = 0.1 # 10%

# Logarithmic decay function for learning rate
def exp_decay(epoch: int, learning_rate: float) -> float:
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

def compile_model(num_classes: int=7, img_size: tuple=(224,224)) -> Sequential:
    
    #dropout_rate = 0.3
    regularization = 0.01 # L1
    learning_rate = 0.00025
    epsilon = 1e-06
    
    activation = activations.relu
    #weight_init = tf.keras.initializers.HeNormal()
    
    padding = "valid"
    # Rescaling is Standartization of the data, this layer will be used with colors 
    # to convert values from 0-255 into 0-1 range. This step is done to normalize data 
    # for Neural Network, key is to have mean value as close to 0 as possible, 
    # this will speed up learning and will provide faster convergence.
    
    
    model = Sequential([
        layers.Rescaling(1./255, input_shape=img_size+(3,)),
         
        layers.Conv2D(16, (3,3), padding=padding, activation=activation,
                      kernel_regularizer=keras.regularizers.l1(regularization)),
                     #kernel_initializer=weight_init),
        layers.MaxPooling2D(),
        
        layers.Conv2D(32, (3,3), padding=padding, activation=activation,
                      kernel_regularizer=keras.regularizers.l1(regularization)),
                     #kernel_initializer=weight_init),
        layers.MaxPooling2D(),
        
        #layers.Dropout(dropout_rate),
        
        layers.Conv2D(64, (3,3), padding=padding, activation=activation,
                      kernel_regularizer=keras.regularizers.l1(regularization)),
                     #kernel_initializer=weight_init),
        layers.MaxPooling2D(),
        
        #layers.Dropout(dropout_rate),
        #layers.BatchNormalization(), # momentum=0.8 
        
        layers.Flatten(),
        
        #layers.GlobalAveragePooling2D(),
    
        #layers.Dropout(rate=dropout_rate),
        #layers.BatchNormalization(),

        layers.Dense(512, activation="relu"), #, kernel_initializer=weight_init),
        #layers.Dropout(rate=dropout_rate),
        layers.Dense(num_classes, activation="softmax") # default linear activation
    ])
    
    # Try focal loss for uneven class distribution
    # Adagrad - Adaptive Gradient
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)     
    
    #import functools 

    #top3acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    #top3acc.__name__ = "top3acc"
    
    # CategoricalCrossentropy is used because labels are OneHotEncoder format
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]) #, "top_k_categorical_accuracy", top3acc])
    return model

## Mish Activation Function
def mish(x):
    return keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

def print_data_distribution(iter_data: keras.preprocessing.image.DirectoryIterator):
    
    # combine data from iterator
    labels_all = iter_data.labels
    n_samples = len(labels_all)   # sum(1 for _ in iter)
    
    class_index = iter_data.class_indices
    
    # count occurences of each class label
    for index, value in class_index.items():
        count = np.count_nonzero(labels_all==value)
        perc = round(count/n_samples*100,2)
        print("{:6}{:8}{:8}%".format(index, count, perc))



def fit_model(model: Sequential, train_ds: Dataset, test_ds: Dataset, 
              epochs: int=10, augment: str="no_augment", verbose: bool=True) -> None:
    
    #trigger = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", min_delta=0.001,
    #                                  patience=10, restore_best_weights=True)
    '''
    early_stop = callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01,
                                      patience=5, restore_best_weights=True)
    save_weights = callbacks.ModelCheckpoint(
                "/output/checkpoints/chkp_EP_{epoch:02d}_VL_{val_loss:.2f}.hdf5", 
                save_best_only=True, monitor="val_loss", mode="min", min_delta=0.01)
    
    lr_sheduler = callbacks.LearningRateScheduler(exp_decay)
    ''';
    
    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)
                       #callbacks=[lr_sheduler]) 
                        #callbacks=[early_stop, save_weights])
    epochs = len(history.history['loss'])
    print("Epoch Length        : {0}".format(epochs))
    
    if verbose:
        evaluate_model(model, train_ds, test_ds)
        print_training_history(history)
    
    model.save("output\keras_models\skin_classifier_{0}_{1}.keras".format(epochs, augment))

def evaluate_model(model: Sequential, train_ds: Dataset, test_ds: Dataset) -> None:
    acc_value = model.evaluate(train_ds, verbose=0)[1]
    val_acc_value = model.evaluate(test_ds, verbose=0)[1]
    print("Model Accuracy      : {0:.4f}".format(acc_value))
    print("Validation Accuracy : {0:.4f}".format(val_acc_value))
    
def print_training_history(history: History) -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    epochs = len(history.history['loss'])
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.xticks(np.arange(epochs), np.arange(1, epochs+1))
    plt.xlabel("Epochs")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xticks(np.arange(epochs), np.arange(1, epochs+1))
    plt.xlabel("Epochs")
    plt.title("Training and Validation Loss")
    plt.show()
    
def print_filters(model: Sequential, plot_weights: bool=False, 
                  test_img_path: str="input/ISIC_0024891.jpg") -> None:
    global IMG_SIZE
    # Dynamically get convolution layer indexes from model
    layer_indx = [i for i,layer in enumerate(model.layers) if "Conv" in str(layer)]

    for layer_i in layer_indx:
        conv_layer = model.layers[layer_i]
        print("Filters     : {0}".format(conv_layer.filters))
        print("Kernel Size : {0}".format(conv_layer.kernel_size))
        
        # Recreate model with a single layer        
        model_t = keras.Model(inputs=model.inputs, outputs=conv_layer.output)

        # Convert image to array
        img = load_img(test_img_path, target_size=IMG_SIZE)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # plot all maps (filters)
        feature_maps = model_t.predict(img)

        cols = 8
        rows = int(conv_layer.filters/cols)

        for i in range(rows*cols):
            ax = plt.subplot(cols, cols, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, i], cmap='gray')

        # show the figure
        plt.show();

        # Plot Filter Weights
        if plot_weights:
            x1w = conv_layer.get_weights()[0][:,:,0,:]
            for i in range(0,conv_layer.filters):
                ax = plt.subplot(cols, cols, i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(x1w[:,:,i], interpolation="nearest", cmap="gray")
            plt.show();
            
def print_confusion_matrix(model: Sequential, val_image_path: str, 
                           normalize: bool=False, img_size: tuple=(224,224),
                           supress_print: bool=False,
                           data: Dataset=None
                           ) -> pd.DataFrame:

    # Get validation images
    batch_size = 64
    
    if data is None:
        generator = ImageDataGenerator()
        image_ds_validation = generator.flow_from_directory(
                directory=val_image_path,
                target_size=img_size,
                batch_size=batch_size,
                shuffle=False # maintains the order to match labels with predictions
        )
    else:
        image_ds_validation = data

    class_names = image_ds_validation.class_indices

    # Get actual y (Truth)
    y = image_ds_validation.labels

    # Make predictions using existing model
    y_pred = model.predict(image_ds_validation)
    # Convert arrays of predictions into integers
    y_pred_int = np.argmax(y_pred, axis=1)

    # Display Confusion Matrix (normalized and not)
    cm = confusion_matrix(y, y_pred_int, normalize=("true" if normalize else None) )
    
    if not supress_print:
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot();

        # Here Recall is Sensitivity
        print("Accuracy Score : ", round(accuracy_score(y, y_pred_int),4))
    # Replaced by DataFrame with added Specificity
    #print("Classification Report :\n", classification_report(y, y_pred_int))

    # Get data from classification report and parse it
    cr = classification_report(y, y_pred_int, digits=4, zero_division=0)
    cr = cr[:cr.index("accuracy")]

    # Remove unwanted spaces 
    while cr.find("  ") > -1:
        cr = cr.replace("  "," ")
    cr = "\n".join([row.strip() for row in cr.split("\n") if row.strip()])

    # Create Pandas DataFrame from String
    c_report = StringIO(cr)
    df = pd.read_csv(c_report, sep=" ")

    # Calculate Specificity
    specificity = []
    tn_total = 0 # Total Negatives
    for i in range(len(class_names)):
        tn_total += cm[i,i]

    for i in range(len(class_names)):
        tn = tn_total - cm[i,i] # total True Negatives minus current cell
        fp = sum(cm[:,i]) - cm[i,i] # current column minus current cell
        specificity.append( round(tn/(tn+fp), 4))
    df["specificity"] = specificity

    # Change columns name and order
    # df.rename(columns={"recall":"sensitivity","support":"total"}, inplace=True)
    columns = df.columns.tolist()
    columns = columns[0:2] + columns[4:] + columns[2:4]
    df = df[columns]

    df["label"] = class_names
    df.set_index("label", drop=True, inplace=True)

    return df

def get_top_k_accuracy(Y_real: np.ndarray,
                       Y_predicted: np.ndarray,
                       k: int=3) -> float:
    metrics = tf.keras.metrics.top_k_categorical_accuracy(
        Y_real, Y_predicted, k=k)
    _, counts = np.unique(metrics, return_counts=True)
    return round(counts[1] / len(Y_real), 4)


    
def print_top_k_accuracy(keras_model: Sequential, 
                         data: keras.preprocessing.image.DirectoryIterator,
                         k: int=5) -> None:
    # OneHotEncoder convertion 
    y_real = to_categorical(data.labels)
    y_pred = keras_model.predict(data)

    for i in range(1,k+1):
        print("Top {0} accuracy : {1}".format(
                i, get_top_k_accuracy(y_real, y_pred, i) ))


        
