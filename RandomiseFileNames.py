import numpy as np
import random

num_points=25
base_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/0012310/06_segmented_tracts/Inverse_tracks/'
train_file_name=base_dir + 'training_set_new_' +str(num_points) + '.txt'
test_file_name =base_dir + 'testing_set_new_' +str(num_points) + '.txt'
validation_file_name = base_dir + 'validation_set_new_' +str(num_points) + '.txt'

f_train = open(train_file_name, 'r')
lines_train=f_train.readlines()
for i in range(0,100):
    random.shuffle(lines_train)

f_test = open(test_file_name, 'r')
lines_test=f_test.readlines()
for i in range(0,100):
    random.shuffle(lines_test)

f_validation = open(validation_file_name, 'r')
lines_validation=f_validation.readlines()
for i in range(0,100):
    random.shuffle(lines_validation)

train_file_name_ran=base_dir + 'training_set_randomised_new_' +str(num_points) + '.txt'
test_file_name_ran =base_dir + 'testing_set_randomised_new_' +str(num_points) +  '.txt'
validation_file_name_ran = base_dir + 'validation_set_randomised_new_' +str(num_points) + '.txt'

f_train_rand = open(train_file_name_ran, 'w')
for lines in lines_train:
    f_train_rand.write(lines)
f_train_rand.close()

f_test_rand = open(test_file_name_ran, 'w')
for lines in lines_test:
    f_test_rand.write(lines)
f_test_rand.close()

f_validation_rand = open(validation_file_name_ran, 'w')
for lines in lines_validation:
    f_validation_rand.write(lines)
f_validation_rand.close()


