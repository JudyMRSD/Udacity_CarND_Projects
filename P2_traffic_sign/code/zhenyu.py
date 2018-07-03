# This is not Jin's code!
# https://github.com/z78406/Traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb
# Load all the packages
import pickle
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

plt.switch_backend('agg')

# TODO: Fill this in based on where you saved the training and testing data
data_dir = "../data/"
ground_truth = data_dir+'dataset_groundtruth/'

training_file = ground_truth + 'train.p'
validation_file=ground_truth + 'valid.p'
testing_file = ground_truth+'test.p'



with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
#X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### split up the training and validation test (since the data has already included validation set)
RATIO = 0.2
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=RATIO)



### shuffle the data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_validation, y_validation = shuffle(X_validation, y_validation)


print("Training Set:   {} samples".format(len(X_train)))
#print("Valid Set:   {} samples".format(len(X_valid)))
print("Test Set:   {} samples".format(len(X_test)))



### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_validation)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = train['features'][0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0]


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.

### Visualize the sign types in X_train (show the first traffic sign image in each category list)
fig,ax_training = plt.subplots(3,15,figsize=(20,10))
fig.suptitle('ALL THE TRAFFIC SIGNS TYPE IN THE TRAINING SET',fontsize = 18) 
for id,ax in enumerate(ax_training.flatten()):
    if id < n_classes:
        class_X = X_train[y_train == id]
        class_img = class_X[1]
        ax.imshow(class_img)
        ax.set_title('{:02d}'.format(id))
    else:
        ax.axis('off')
plt.setp([a.get_xticklabels() for a in ax_training.flatten()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_training.flatten()], visible=False)   
plt.draw



## plot the bar of each traffic signs
train_list = np.zeros(n_classes)
test_list = np.zeros(n_classes)
for type in range(n_classes):
    train_list[type] = np.sum(y_train == type)
    test_list[type]  = np.sum(y_test  == type)
    
bar_width = 0.5    
x_pos_train = np.arange(n_classes) 
x_pos_test  = np.arange(n_classes) 
train_bar = plt.bar(x_pos_train, train_list,  width = 0.3, color = 'r', label = 'training set' )
test_bar  = plt.bar(x_pos_test + bar_width, test_list,  width = 0.3, color = 'g', label = 'testing set')
plt.xticks(np.arange(0, n_classes, 5) + bar_width)
plt.xlabel('Traffic signs')
plt.ylabel('Traffic sign numbers')
plt.title('Traffic sign type and numbers')
plt.legend()
plt.tight_layout()
plt.show()



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.



### normalize the data
def pre_process(data):
    
    data = data.astype('float32')
    data = (data - 128.) / 128.
    return data
train_data = pre_process(X_train)
test_data  = pre_process(X_test)


### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten




def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.leaky_relu(fc1)
    
    # SOLUTION: DROPOUT
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.sigmoid(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

### set up tensorflow

import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 128


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.



### set up the placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

### training pipeline
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
weight = tf.train.Saver()
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### initiate the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        
        train_accuracy      = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        weight.save(sess, save_path = './tmp/model.ckpt')
        print()
        

    print("Model saved")


### Test the model
with tf.Session() as sess:
    weight.restore(sess, './tmp/model.ckpt')
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))




