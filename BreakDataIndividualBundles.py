import numpy as np
import tensorflow as tf
import random
import os, os.path

train_percentage = 0.8
base_name_fiber='Fiber_'

base_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/06_segmented_tracts/Inverse_tracks/'

trk_file=base_dir + 'track_list_all.txt'
text_trk_file = open(trk_file, "r")
trk_labels = text_trk_file.readlines()
count =0
for trk_i in trk_labels:
    trk_i=trk_i.strip(' \n')
    trk_labels[count] = trk_i
    count = count +1
num_fiber_bundles = len(trk_labels)
print(trk_labels)

num_points=25

#data_dir='/Users/vgupta/PycharmProjects/DeepLearning_tractography/06_segmented_tracts/06_segmented_tracts/'

training_file_name= base_dir + 'training_set_new_' + str(num_points) + '.txt'
testing_file_name= base_dir + 'testing_set_new_' + str(num_points) + '.txt'
validation_file_name = base_dir + 'validation_set_new_' + str(num_points) +'.txt'

f_train = open(training_file_name, 'w')
f_test = open(testing_file_name, 'w')
f_valid = open(validation_file_name, 'w')


for trk_i in trk_labels:
    print(trk_i)
    directory = base_dir + trk_i + '_' + str(num_points) + '_inv'
    number_of_files = len([item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))])
    number_training_set_i = int(train_percentage*number_of_files)

    for i in range(1, number_training_set_i +1 ):
        file_name_fiber = directory + '/' + 'Fiber_' + str(i) + '.txt'
        f_train.write(file_name_fiber)
        f_train.write('\n')

    remaining_files = number_of_files - number_training_set_i
    number_testing_set = int(remaining_files/2)

    for i in range(number_training_set_i+1, (number_testing_set+number_training_set_i+1)):
        file_name_fiber_test = directory + '/' + 'Fiber_' + str(i) + '.txt'
        f_test.write(file_name_fiber_test)
        f_test.write('\n')

    start_idx_valid=number_training_set_i + number_testing_set +1

    for i in range(start_idx_valid, (number_of_files)):
        file_name_fiber_valid = directory + '/' + 'Fiber_' + str(i) + '.txt'
        f_valid.write(file_name_fiber_valid)
        f_valid.write('\n')



f_train.close()
f_test.close()
f_valid.close()
