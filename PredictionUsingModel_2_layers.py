import numpy as np
import tensorflow as tf
import GeneratePredictionBatch as gp

subj_model = '0012310'
subj = '0035019'

num_points=25
#data_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/Inverse_tracks/' + str(num_points) + '_points_resampled/'
data_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/' + str(num_points) + '_points_resampled/'
#validation_file= data_dir + 'testing_set_randomised_new_25.txt'
validation_file= data_dir + 'validation_set_randomised_new_25.txt'
#validation_file= data_dir + 'all_fibers_new_25.txt'
#validation_file= data_dir + 'training_set_randomised_new_25.txt'
validation_file_fiber_names = open(validation_file, 'r')
validation_file_list = validation_file_fiber_names.readlines()

print(validation_file)
with open(validation_file) as f:
    num_valid_set = sum(1 for _ in f)

# Network Parameters
n_hidden_1 = 50 # 1st layer number of features
n_hidden_2 = 50 # 2nd layer number of features
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
model_file_base='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/'+ subj_model +'/Trained_models_corrected/25_points/Inverse_tracks/2_layers/'  + str(n_hidden_1)+ '_' + str(n_hidden_2) + '_' + 'Batch_size_' + str(batch_size)

genPredictionBatch = gp.GeneratePredictionBatch()
batch_pred = num_valid_set
#batch_pred = 20

#Create the dictionary
trk_file='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/06_segmented_tracts/track_list_all.txt'
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
#num_batches = 100
init = tf.initialize_all_variables()

prediction_file_name=data_dir + 'Prediction_all_fiber_file.txt'
prediction_file=open(prediction_file_name, 'w')


print(model_file_base)    

print ("Num Valid Set: ", num_valid_set)
totalBatch = int(num_valid_set/batch_size)
print("Total Batch: ", totalBatch)

with tf.Session() as sess:
    sess.run(init)
    ckpt_dir = tf.train.get_checkpoint_state(model_file_base)
    saver.restore(sess, ckpt_dir.model_checkpoint_path)
    print('Model loaded: ', ckpt_dir.model_checkpoint_path)
    total_batch = int(num_valid_set/batch_size)
    prediction_numpy_array=np.empty([num_valid_set], dtype='int' )
    for i in range(total_batch+1):
       genValidationBatch = gp.GeneratePredictionBatch()
       prediction=tf.argmax(pred,1)
       start_idx = i*batch_size
       end_idx = start_idx + batch_size
       if (end_idx > num_valid_set):
          end_idx = num_valid_set
       print ("Start Idx: ", start_idx)
       print ('End Idx: ', end_idx)

       batch_xs = genValidationBatch.CreatePredictionBatch(validation_file_list, start_idx, end_idx)
       print (type(batch_xs))
       print(batch_xs.shape)
       #predict_result = prediction.eval(feed_dict={x: batch_xs})
       predict_result = prediction.eval(feed_dict={x: batch_xs})
       print(predict_result.shape)
       prediction_numpy_array[start_idx:end_idx] = predict_result 

prediction_numpy_vector = prediction_numpy_array.flatten()

print ( prediction_numpy_array.shape)
prediction_list = prediction_numpy_vector.tolist()
print(type(prediction_list))

with prediction_file as f:
   for s in prediction_list:
      f.write(str(s) + '\n')

