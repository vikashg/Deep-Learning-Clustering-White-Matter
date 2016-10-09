from __future__ import print_function
import tensorflow as tf
import numpy as np
import GenerateBatches_25 as gb
import os, os.path

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

## Get the training and test data
#Read Files
num_points=25
base_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/06_segmented_tracts/25_points_resampled/'

training_fileList_randomized=base_dir + 'training_set_randomised_new_' + str(num_points) + '.txt'
training_file_fiber_names = open(training_fileList_randomized, "r")
train_name_list = training_file_fiber_names.readlines()

testing_fileList_randomized=base_dir + 'testing_set_randomised_new_' + str(num_points)+ '.txt'
testing_file_fiber_names = open(testing_fileList_randomized, "r")
test_name_list = testing_file_fiber_names.readlines()


with open(training_fileList_randomized) as f:
    num_train_set = sum(1 for _ in f)


with open(testing_fileList_randomized) as f:
    num_test_set = sum(1 for _ in f)

## Divide Train Set Test Set and Validation Set
## Make track label lookup table
trk_file=base_dir + 'track_list_all.txt'
text_trk_file = open(trk_file, "r")
trk_labels = text_trk_file.readlines()

count =0
for trk_i in trk_labels:
    trk_i=trk_i.strip(' \n')
    trk_labels[count] = trk_i
    count = count +1

num_fiber_bundles = len(trk_labels)

keys = list(range(1, num_fiber_bundles+1, 1))
Track_label_lookup=dict(zip(trk_labels, keys))


########### Now the code for the multilayer perceptron
# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
n_hidden_2 = 128 # 2nd layer number of features
n_input = 3*num_points # number pf features
n_classes = num_fiber_bundles # MNIST total classes (0-9 digits)

# Learning parameters
# Parameters
learning_rate = 0.00001
training_epochs = 200
batch_size = 1000*3
display_step = 1

model_file_base='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/Trained_models/25_points/Inverse_tracks/' + 'Epoch_' + str(training_epochs) + '/2_layers/' + str(n_hidden_1) + '_' + str(n_hidden_2) + '/'

if not os.path.exists(model_file_base):
    os.makedirs(model_file_base)

output_file= model_file_base + 'Output_file_' + '2_layers_' + str(n_hidden_1) + '_' + str(n_hidden_2) + '.txt'
f = open(output_file, 'w')
f.write('Learning Rate: '+ str(learning_rate) + '\n')
f.write('Training Epochs: ' + str( training_epochs) + '\n')
f.write('Hidden Layers: 2 ' +'\n') 
f.write('Hidden Layers 1: ' + str( n_hidden_1) + '\n' )
f.write('Hidden Layers 2: ' + str( n_hidden_2) + '\n' )
f.write('ReLu Activation' + '\n')

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver() ## For saving the model

f.write('Results saved in: ' + str( model_file_base) + '\n')

print('Everything check A ok' + '\n')
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = int(num_train_set*3/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            testClass = gb.GenerateBatches()
            batch_xs, batch_ys = testClass.CreateBatches_XY(num_train_set, train_name_list, batch_size, num_fiber_bundles, Track_label_lookup)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
        if epoch % display_step == 0:
            model_file_path_i = model_file_base + 'MLP_epoch_' + str(epoch) + '.cpkt'
            save_path = saver.save(sess, model_file_path_i)
            print("Model saved in file: %s" % save_path)
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            output_line = 'Epoch: ' + str(epoch+1) + 'cost= '+ str(avg_cost) + '\n'
#            f.write("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            f.write(output_line)

    print("Optimization Finished!")
    f.write("Optimization Finished!")
    model_file_path=model_file_base + 'Trained_model_' + str(training_epochs) + '_' + str(learning_rate) + '.cpkt'
    save_path = saver.save(sess, model_file_path)
    print("Model saved in file: %s" % save_path)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    batch_x_test, batch_y_test = testClass.CreateBatches_XY(num_test_set, test_name_list, num_test_set, num_fiber_bundles,
                                                            Track_label_lookup)

    accuracy_eval = accuracy.eval({x: batch_x_test, y: batch_y_test})
    print("Accuracy:", accuracy_eval)
    accuracy_line = 'Accuracy: ' + str(accuracy_eval)
    f.write(accuracy_line)

f.close()
