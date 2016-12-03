import numpy as np
import random

num_points=100
subj='0112920'
base_dir='/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/Outputs/' + subj + '/06_segmented_tracts/' + str(num_points) + '_points_resampled/' 

train_file_name=base_dir + 'AllFibers_w_cc.txt'

f_train = open(train_file_name, 'r')
lines_train=f_train.readlines()
for i in range(0,200):
    random.shuffle(lines_train)

train_file_name_ran=base_dir + 'AllFibers_randomised_w_cc_100.txt'

f_train_rand = open(train_file_name_ran, 'w')
for lines in lines_train:
    f_train_rand.write(lines)
f_train_rand.close()



