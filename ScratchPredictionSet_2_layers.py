import numpy as np
import numpy as np
import tensorflow as tf
import GeneratePredictionBatch as gp
import GenerateBatches_newSubj as gb

subj_model = '0012310'
subj = '0112920'

num_points=100
#data_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/Inverse_tracks/' + str(num_points) + '_points_resampled/'
data_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/' + str(num_points) + '_points_resampled/'
validation_file= data_dir + 'testing_set_randomised_new_100.txt'
#validation_file= data_dir + 'validation_set_randomised_new_100.txt'
#validation_file= data_dir + 'training_set_randomised_new_25.txt'
validation_file_fiber_names = open(validation_file, 'r')
validation_file_list = validation_file_fiber_names.readlines()

print(validation_file)
with open(validation_file) as f:
    num_valid_set = sum(1 for _ in f)

# Network Parameters
n_hidden_1 = 150 # 1st layer number of features
n_hidden_2 = 150 # 2nd layer number of features
n_input = 3*num_points # number pf features
n_classes = 21 # MNIST total classes (0-9 digits)

# Learning parameters
# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 1000*3
display_step = 1

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


saver = tf.train.Saver() ## For saving the model
model_file_base='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/'+ subj_model +'/Trained_models_corrected/' + str(num_points) + '_points/Inverse_tracks/2_layers/'  + str(n_hidden_1)+ '_' + str(n_hidden_2) + '_' + 'Batch_size_' + str(batch_size)

genPredictionBatch = gp.GeneratePredictionBatch()
batch_pred = num_valid_set
#batch_pred = 20

#Create the dictionary
trk_file='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/06_segmented_tracts/track_list_all.txt'
#trk_file='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0035019/06_segmented_tracts/100_points_resampled/track_list_all.txt'
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

print("Num of Fibers in Validation Set: " , num_valid_set )
num_batches = num_valid_set;
#num_batches = 1
init = tf.initialize_all_variables()

print(model_file_base)    
with tf.Session() as sess:
    sess.run(init)
    ckpt_dir = tf.train.get_checkpoint_state(model_file_base)
    saver.restore(sess, ckpt_dir.model_checkpoint_path)
#    saver.restore(sess, model_name)
    print('Model loaded: ', ckpt_dir.model_checkpoint_path)

    print('Num hidden: ', n_hidden_1 )
    genValidationBatch = gb.GenerateBatches()
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    batch_xs, batch_ys = genValidationBatch.CreateBatches_XY(num_batches, validation_file_list, batch_pred, num_fiber_bundles, Track_label_lookup, num_points)
    print("Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))
