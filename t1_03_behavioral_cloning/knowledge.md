# Keras

## Sequential Model

*(Using Keras v1.2.1 from the starter kit? See the archived documentation for `keras.models.Sequential` [here](https://faroit.github.io/keras-docs/1.2.1/models/sequential/).)

### Neuronal Networks, incl. Convolutions, Pooling,  Dropout and Testing (evaluation)

#### Quiz

Multi-layer feedforward neural network to classify traffic sign images using Keras.

See Keras documentation about models and layers. The Keras example of a [Multi-Layer Perceptron](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py) network is similar to what you need to do here. Use that as a guide, but keep in mind that there are a number of differences.

##### Solution

```
# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Build the Fully Connected Neural Network in Keras Here
model = Sequential()

# Layer: Convolutional layer with 32 filters, a 3x3 kernel, and valid padding
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))

# Layer: Polling layer with 2x2 dimentions
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='default'))

# Layer: Dropout. The rate specified for dropout in Keras is the opposite of TensorFlow! 
#TensorFlow uses the probability to keep nodes, while Keras uses the probability to drop them.
model.add(Dropout(0.5))

# Layer: ReLU activation
model.add(Activation('relu'))

# Layer: Flatten() with the input_shape set to (32, 32, 3).
model.add(Flatten(input_shape=(32, 32, 3)))

# Layer: Dense() with an output width of 128.
model.add(Dense(128))

# Layer: ReLU activation function
model.add(Activation('relu'))

# Layer: fully connected layer. Set the output layer width to 5, because for this data set there are only 5 classes.
model.add(Dense(5))

# Layer: Use a softmax activation function after the output layer.
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

# TODO: Evaluate the test data in Keras Here
metrics = model.evaluate(X_normalized_test, y_one_hot_test, batch_size=32, verbose=1, sample_weight=None)
# TODO: UNCOMMENT CODE
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
```



## Transfer learning

### The Four Main Cases When Using Transfer Learning

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

- the size of the new data set, and
- the similarity of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

1. new data set is small, new data is similar to original training data
2. new data set is small, new data is different from original training data
3. new data set is large, new data is similar to original training data
4. new data set is large, new data is different from original training data

<img src="https://video.udacity-data.com/topher/2017/February/58a608ea_02-guide-how-transfer-learning-v3-01/02-guide-how-transfer-learning-v3-01.png" alt="img" style="zoom:50%;" />![img](https://video.udacity-data.com/topher/2017/February/58a73c8d_02-guide-how-transfer-learning-v3-04/02-guide-how-transfer-learning-v3-04.png)

![img](https://video.udacity-data.com/topher/2017/February/58a73c8d_02-guide-how-transfer-learning-v3-04/02-guide-how-transfer-learning-v3-04.png)![img](https://video.udacity-data.com/topher/2017/February/58a73bd8_02-guide-how-transfer-learning-v3-06/02-guide-how-transfer-learning-v3-06.png)

![img](https://video.udacity-data.com/topher/2017/February/58a73ccd_02-guide-how-transfer-learning-v3-08/02-guide-how-transfer-learning-v3-08.png)![img](https://video.udacity-data.com/topher/2017/February/58a73d0d_02-guide-how-transfer-learning-v3-10/02-guide-how-transfer-learning-v3-10.png)

### How to apply transfer learning

Two popular methods are **feature extraction** and **finetuning**.

1. **Feature extraction**. Take a pretrained neural network and replace the final (classification) layer with a new classification layer, or perhaps even a small feedforward network that ends with a new classification layer. During training the weights in all the pre-trained layers are frozen, so only the weights for the new layer(s) are trained. In other words, the gradient doesn't flow backwards past the first new layer.
2. **Finetuning**. This is similar to feature extraction except the pre-trained weights aren't frozen. The network is trained end-to-end.

#### Feature extraction example (incl. training)

```python
import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128

with open('./train.p', 'rb') as f:
    data = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
init_op = tf.initialize_all_variables()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
```

## Conclusions

Consider **feature** extraction when ...

... the new dataset is small and similar to the original dataset. The higher-level features learned from the original dataset should transfer well to the new dataset.

Consider **finetuning** when ...

... the new dataset is large and similar to the original dataset. Altering the original weights should be safe because the network is unlikely to overfit the new, large dataset.

... the new dataset is small and very different from the original dataset. You could also make the case for training from scratch. If you choose to finetune, it might be a good idea to only use features from the first few layers of the pre-trained network; features from the final layers of the pre-trained network might be too specific to the original dataset.

Consider **training from scratch** when ...

... the dataset is large and very different from the original dataset. In this case we have enough data to confidently train from scratch. However, even in this case it might be beneficial to initialize the entire network with pretrained weights and finetune it on the new dataset.

Finally, keep in mind that for a lot of problems you won't need an architecture as complicated and powerful as VGG, Inception, or ResNet. These architectures were made for the task of classifying thousands of complex classes. A smaller network might be a better fit for a smaller problem, especially if you can comfortably train it on moderate hardware.

# Project

## Start env

source activate carnd-term1

