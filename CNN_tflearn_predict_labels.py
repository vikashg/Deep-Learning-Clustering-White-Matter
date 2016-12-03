from __future__ import division, print_function, absolute_import
import os, os.path
import GenerateBatches_predict_1 as gb 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np

num_points = 100
subj = '0012310'

subj_pred='0112920'

base_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/Inverse_tracks/' + str(num_points) + '_points_resampled/'

pred_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj_pred + '/06_segmented_tracts/' + str(num_points) + '_points_resampled/'


part='aa'

testing_fileList_randomized=pred_dir + 'AllFibers_randomised_wo_cc_100_part_' + part + '.txt'
#testing_fileList_randomized=pred_dir + 'temp.txt'
testing_file_fiber_names = open(testing_fileList_randomized, "r")
test_name_list = testing_file_fiber_names.readlines()


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

## New code for DNN
network = input_data(shape=[None, num_points, 3, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
#network = max_pool_2d(network, 2)

network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer='L2')
#network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, num_fiber_bundles, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')

#Generating 

testClass = gb.GenerateBatches()
#batch_xs_train, batch_ys_train = testClass.CreateBatches_XY(num_train_set, train_name_list, batch_size, num_fiber_bundles, Track_label_lookup, num_points)

#batch_X_train = batch_xs_train.reshape([-1, num_points, 3, 1])

batch_size_test = num_test_set
#batch_size_test = 1300
batch_xs_test, batch_ys_test = testClass.CreateBatches_XY(num_test_set, test_name_list, batch_size_test, num_fiber_bundles, Track_label_lookup, num_points)

print('Batch formed')
batch_X_test = batch_xs_test.reshape([-1, num_points, 3, 1])

output_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/Trained_models_CNN/' + str(num_points) + '_points_resampled/'

#Training data
num_epoch = 10
model = tflearn.DNN(network, tensorboard_verbose=0)

model_name=output_dir + 'CNN_conv2d_32_conv2d_64_fc_128_fc_256_epoch_' + str(num_epoch) + '.tflearn'

model.load(model_name)
print ('Model loaded: ', model_name)
a = model.predict(batch_X_test)
b=np.asarray(a)
predict_val = np.argmax(b, axis = 1)
predict_val.astype(int)
predicted_values_file = pred_dir + 'PredictedValues_wo_cc_frontal_' + part + '.txt'
np.savetxt(predicted_values_file, predict_val, fmt='%d')

temp_eval = model.evaluate(batch_X_test, batch_ys_test)
print(temp_eval)
