# -*- coding: utf-8 -*-
"""
"""
"""
Changes done:
1. Changed train/test/valid split
2. Changed number of input classes from 10 to 2 ( for Y and dense to one hot )

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2



# convert grayscale image to array
def image_to_array(img):
    return np.array(img, dtype=np.uint8)
    
# Import Required Libraries here
def ReadCSV(fileName):
    # load file - skip header row
    data = np.loadtxt(fileName, skiprows=1, delimiter=',')
    # store the labels in label array
    label = data[:, 0]
    # store pixel intensities in data array
    data = data[:, 1:].reshape(data.shape[0], -1)
    # normalize data array
    data = data / 255.0 - 0.5
    # Read CSV File
    # define two np arrays named "data" and "label" for MNIST images and labels
    # normalize data by: data = data/255.0 - 0.5
    return data, label


# This function initializes the weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# This function converts the label list to one-hot format - changed num_classes to 2
def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot


# Define 3 layers with 4th layer being the output layer
def model(X):
    # define your NN model here
    layer_1 = tf.nn.relu(tf.matmul(X, w1))
    layer_2 = tf.nn.relu(tf.matmul(layer_1, w2))
    layer_3 = tf.nn.relu(tf.matmul(layer_2, w3))
    output_layer = tf.matmul(layer_3, w4)
    return output_layer
    # and so on


data, labelA = ReadCSV('football_group.csv')

# Check the dimentions before and after converting to one-hot. Make sure it works properly.
label = dense_to_one_hot(labelA)
print("Checking the dataset shapes: pixel data " + str(data.shape), " , one hot encoded labels: " + str(label.shape))

trainSize = 137 # As number of samples are only 177
validSize = 20  # Split used 137/20/20


train_data = data[:trainSize, :]
train_label = label[:trainSize, :]
valid_data = data[trainSize:trainSize + validSize, :]
valid_label = label[trainSize:trainSize + validSize, :]
test_data = data[trainSize + validSize:, :]
test_label = label[trainSize + validSize:, :]

# Define Placeholders:
X = tf.placeholder("float", [None, 2352])
Y = tf.placeholder("float", [None, 2]) # changed num classes to 2

# define number of columns in input data for layer 1
input_cols = data.shape[1]

# define hidden units in each layer
hidden_layer_1 = 512
hidden_layer_2 = 256
hidden_layer_3 = 64

# define number of classes in output data for last layer
output_cols = label.shape[1]

# Define and initialize weights:
w1 = init_weights([input_cols, hidden_layer_1])
w2 = init_weights([hidden_layer_1, hidden_layer_2])
w3 = init_weights([hidden_layer_2, hidden_layer_3])
w4 = init_weights([hidden_layer_3, output_cols])
# and so on for w2, w3, ...

output = model(X)

# define learning rate
learning_rate = 0.0001

# define loss - softmax is applied to the output and crossentropy is calculated
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))

# Minimize loss using the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Separately apply softmax for accuracy calculation
output = tf.nn.softmax(output)

# Extract predicted class from the output class probabilities obtained in previous step
predicter = tf.argmax(output, 1)

# run and initialize the TF session
batchSize = 16 # changed batch size to 16

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

losses = []
validAcc = []

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for iteration in range(2):  # no. of epochs is 2
        for i in range(0, trainSize, batchSize):
            print("Iteration: " + str(iteration) + " *** Batch: " + str(i))

            batch_x, batch_y = train_data[i:(i + batchSize), :], train_label[i:(i + batchSize),
                                                                 :]  # make x, y batches using batchsize

            _, loss_value = sess.run([optimizer, loss],
                                     feed_dict={X: batch_x, Y: batch_y})  # run optimizer and obtain loss value

            losses.append(loss_value)  # store loss value in a list

            prediction = sess.run(predicter,
                                  feed_dict={X: valid_data, Y: valid_label})  # obtain predictions for validation data

            correct_labels = np.argmax(valid_label,
                                       axis=1)  # obtain correct labels from one hot encoded validation set labels

            valid_accuracy = np.mean((prediction == correct_labels)) * 100  # calculate validation accuracy

            validAcc.append(valid_accuracy)  # store validation accuracy in valid acc array

            print("Step: " + str(i / batchSize) + " *** Loss: " + str(np.round(loss_value, 4)) + " *** Validation accuracy: ",
                  str(np.round(valid_accuracy, 1)) + "%")
            # run the session on optimizer by feeding the minibatches of placeholders
            # run the session on predicter by feeding the validation data
            # append the current loss value to the losses list
            # append the validattion accuracy to the validAcc list

    print("*" * 22 + "Finished Training" + "*" * 22)

    prediction = sess.run(predicter, feed_dict={X: test_data, Y: test_label})  # run session using test data as feed
    print(prediction)
    correct_labels = np.argmax(test_label, axis=1)  # obtain test labels

    test_accuracy = np.mean((prediction == correct_labels)) * 100  # calculate test accuracy

    print("Final accuracy: " + str(np.round(test_accuracy, 2)) + "%")

    # Plot losses vs training step during training
    plt.figure()
    plt.plot(losses)
    plt.title("Training losses")
    plt.ylabel('Loss')
    plt.show()

    # Plot validation accuracy vs training step during training
    plt.figure()
    plt.plot(validAcc)
    plt.title("Validation Accuracy")
    plt.ylabel('Accuracy')
    plt.show()

    print("Confusion matrix for the test set")
    print(pd.crosstab(correct_labels, prediction, rownames=['Actual'], colnames=['Predicted'], margins=True))


#display result

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}

    cap = cv2.VideoCapture("/home/jite/Downloads/ManU.mp4")

    counter = 0
    while 1:
        ret, img = cap.read()
        (rects, weights) = hog.detectMultiScale(img, **hogParams)            

        for (x, y, w, h) in rects:
            
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))
            cv2.imwrite('/home/jite/Downloads/video2/video'+str(counter)+'.png',roi)
            
            images = []
            images.append(roi.ravel())
            y_test_images = np.zeros((1, 2)) 
            prediction = sess.run(predicter, feed_dict={X: images, Y: y_test_images})  # run session ROI as feed
            print(prediction)

            if prediction[0] < 1:
                cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(img,'Manu',(x + w, y + h), font, 0.7,(200,200,255),1,cv2.LINE_AA)
                cv2.putText(img,'0% confidence',(x + w, y), font, 0.7,(200,200,255),1,cv2.LINE_AA)
            else:
                cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(img,'Chelsea',(x + w, y + h), font, 0.7,(200,200,255),1,cv2.LINE_AA)
                cv2.putText(img,'0% confidence',(x + w, y), font, 0.7,(200,200,255),1,cv2.LINE_AA)
                
                

        cv2.imshow('img',img)
        counter = counter + 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# test the model on the test_data
# report final Accuracy rate

# plot the losses
# plot the validation accuracy rates
# Show confusion matrix (predicted vs. desired)
