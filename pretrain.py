import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
from model import *
from glob import glob
from tensorflow.keras import *

print("Starting PreTraining Script")
print("Tensorflow version:", tf.__version__)

# Variables: 

batch_size = 2
num_epochs = 10
optimizer = optimizers.Adam
class_weight_factor = 0.05
boundary_weight_factor = 10.0
calculate_categorical_accuracy = True
glob_learning_rate = 1e-5
using_checkpoint = True

# class_weight_factor influences how much importance is given to the categorically weighted loss
# vs the unweighted loss. 0 means unweighted, 1 is all categorically weighted
if ((class_weight_factor < 0.0) or (class_weight_factor > 1.0)):
    sys.exit("class_weight_factor must be a number between 0 and 1 inclusive.")

# boundary weighting factor multiplies the additional boundary loss
if (boundary_weight_factor < 0.0):
    sys.exit("boundary_weight_factor must be a positive number.")

now = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
print("Current time: " + now)
loss_output_file = "/proj/NIRAL/users/nic98/models/unet_4/logfiles/loss_record_" + now + ".csv"
model_checkpoint_save_path = '/proj/NIRAL/users/nic98/models/unet_4/checkpoints/parameters_' + now

# Supply parent weights
parent_time = "10-03-2020-12:17:30"
if (not using_checkpoint):
    parent_time = "None"
model_checkpoint_load_path = "/proj/NIRAL/users/nic98/models/unet_4/checkpoints/parameters_" + parent_time
if (using_checkpoint):
    print("parent training time: " + parent_time)

# get files needed for training
data_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/data_arrays/*"))
label_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/label_arrays/*"))
weight_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/weight_arrays/*"))
boundary_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/boundary_arrays/*"))

# Check we have admissible batch size
if ((len(data_files) % batch_size) != 0):
    print("batch size does not divide number of features")
    print("num features: ", len(data_files), "batch size: ", batch_size)
    quit()


record_file = open("/proj/NIRAL/users/nic98/models/unet_4/training_record.log", "a")
record_file.write("Continuing training from time " + parent_time + " for {} epochs at ".format(num_epochs) + now + "\n")
record_file.close()

# Get model
model = unet_4()

if (using_checkpoint):
    print("loading checkpoints from " + model_checkpoint_load_path)
    model.load_weights(model_checkpoint_load_path)

def train(model, learning_rate, optimizer, features, labels, weights = tf.constant(1.0), boundaries = tf.constant(0.0),
        class_weights_factor = tf.constant(1.0), boundary_weights_factor = tf.constant(0.0)):
    optimizer = optimizer(learning_rate=learning_rate)

    #Weights is the volumetric weights
    #class_weights_factor is the volumetric weighting factor
    #Compute loss and do gradient descent
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        prediction = model(features)
        prediction = tf.math.add(prediction, tf.constant(1e-5))
        loss_tensor = tf.reduce_sum(tf.math.multiply(labels, tf.math.log(prediction)), axis=-1)
        # The following line is equivalent to:
        # loss_tensor = class_weights_factor * loss_tensor * weights + (1 - class_weights_factor) * loss_tensor
        loss_tensor = tf.add(tf.math.multiply(tf.math.multiply(loss_tensor, weights), class_weights_factor), 
                tf.math.multiply(loss_tensor, tf.math.subtract(tf.constant(1.0), class_weights_factor)))
        # Now doing additional boundary loss:
        loss_tensor = tf.add(tf.math.multiply(loss_tensor, tf.math.multiply(boundary_weights_factor, boundaries)), loss_tensor)
        # Last step of computing categorical crossentropy loss:
        loss = tf.math.multiply(tf.constant(-1.0), tf.reduce_sum(loss_tensor))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

print("Training on {} epochs of {} batches each".format(num_epochs, len(data_files) // batch_size))

#Loop over epochs and batches to train
epochs = range(num_epochs)
loss_file = open(loss_output_file, "a")
loss_file.write("Continuing training with boundary loss from session " + parent_time + "\n")
loss_file.write("Epoch, Batch, Loss, Categorical Accuracy \n")
for epoch in epochs:
    permuted_indices = np.random.permutation(len(data_files))
    for batch in range( len(data_files) // batch_size ):
        #Load batch as we need it
        data_arrays = []
        label_arrays = []
        weight_arrays = []
        boundary_arrays = []
        print("Loading Data for epoch", epoch, "batch", batch)
        for i in range(batch_size):
            data_arrays.append(np.load(data_files[permuted_indices[batch*batch_size + i]]))
            label_arrays.append(np.load(label_files[permuted_indices[batch*batch_size + i]]))
            weight_arrays.append(np.load(weight_files[permuted_indices[batch*batch_size + i]]))
            boundary_arrays.append(np.load(boundary_files[permuted_indices[batch*batch_size + i]]))
        
        data_arrays = tf.convert_to_tensor(data_arrays, dtype=tf.float32)
        label_arrays = tf.convert_to_tensor(label_arrays, dtype=tf.float32)
        weight_arrays = tf.convert_to_tensor(weight_arrays, dtype=tf.float32)
        boundary_arrays = tf.convert_to_tensor(boundary_arrays, dtype=tf.float32)

        print("Epoch", epoch, "progress: ", batch/( len(data_arrays) / batch_size))
        loss, prediction = train(model, glob_learning_rate, optimizer, data_arrays, label_arrays, weight_arrays, boundary_arrays, tf.constant(class_weight_factor), tf.constant(boundary_weight_factor))
        cat_acc = None
        if(calculate_categorical_accuracy):
            m = tf.keras.metrics.CategoricalAccuracy()
            _ = m.update_state(label_arrays, prediction)
            cat_acc = m.result().numpy()
        print("LOSS: ", loss, "ACC: ", cat_acc)
        loss_file.write("{}, {}, {}, {}\n".format(epoch, batch, loss, cat_acc))

loss_file.close()

#Save Model Weights:
model.save_weights(model_checkpoint_save_path)
