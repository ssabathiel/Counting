import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['backend'] = "Qt4Agg"




def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)




#Step 1 - Get Input Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = np.load('Object_Sets_Array/trX.npy')
trY = np.load('Object_Sets_Array/trY.npy')
teX = np.load('Object_Sets_Array/teX.npy')
teY = np.load('Object_Sets_Array/teY.npy')

print("Size of Input", trX[0,:].size)
print("Size of Output", trY[0,:].size)



##################################################
##################################################

def conv_net(x_arr, n_classes, dropout, reuse, is_training):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x_arr, shape=[-1, 28, 28, 1])

    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    # Convolution Layer with 64 filters and a kernel size of 3
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv2)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 1024)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(fc1, n_classes)

    return out


##################################################
##################################################






#Step 2 - Create input and output placeholders for data
X = tf.placeholder("float", [None, trX[0,:].size], name="X")
Y = tf.placeholder("float", [None, trY[0,:].size], name="Y")

#Step 3 - Initialize weights
w_h = init_weights([trX[0,:].size, 625], "w_h")
w_h2 = init_weights([625, 625], "w_h2")
w_o = init_weights([625, trY[0,:].size], "w_o")

#Step 4 - Add histogram summaries for weights
tf.summary.histogram("w_h_summ", w_h)
tf.summary.histogram("w_h2_summ", w_h2)
tf.summary.histogram("w_o_summ", w_o)

#Step 5 - Add dropout to input and hidden layers
p_keep_input = tf.placeholder("float", name="p_keep_input")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

#Step 6 - Create Model
#py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
n_classes = trY[0,:].size
py_x = conv_net(X, n_classes, 0.25, reuse=False, is_training=True)

#Step 7 Create cost function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels= Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Add scalar summary for cost tensor
    tf.summary.scalar("cost", cost)

#Step 8 Measure accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)

#first_array = trX
#plt.imshow(first_array)
# Actually displaying the plot if you are not in interactive mode
#plt.show()
#print("hello")
#plt.savefig("fig.png")

#Step 9 Create a session
acc_arr = []
episodes = []
n_episodes = 200
with tf.Session() as sess:
    # Step 10 create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # Step 11 you need to initialize all variables
    tf.initialize_all_variables().run()

    #Step 12 train the  model
    for i in range(n_episodes):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op,                    feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_input: 0.8, p_keep_hidden: 0.5})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)                   # Report the accuracy
        acc_arr = np.append(acc_arr,acc)
        episodes = np.append(episodes, i)

matplotlib.get_backend()
print(acc_arr.shape)
print(episodes.shape)
#print("np.arange(0,n_episodes)", np.arange(0,n_episodes))


plt.plot(np.arange(0,n_episodes), acc_arr)
plt.xlabel('Accuracy')
plt.ylabel('Episodes')
plt.ylim(0,1)
plt.title('Accuracy development during learning to count with a standard NN')
plt.savefig('Standard_NN.png')
#plt.plot([1,2,3], [1,1,1])
plt.show()
