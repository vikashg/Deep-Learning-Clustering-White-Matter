import numpy as np
import random
import tensorflow as tf


class GenerateBatches():
    __placeholder = 0

    def __init__(self):
        _tempPlace =0

    def CreateBatches_XY(self, num_fiber_set, train_name_list, batch_size, num_fibers_bundles  , Track_label_dictionary):
        # lines = open(fileName_FiberList_str).readlines()
        num_points=25
        line_numbers = np.random.randint(0, num_fiber_set - 1, size = batch_size)
        batch_X = np.full((batch_size, 3*num_points), 1.0, dtype = 'float')
        batch_Y= self.Create_batchY(batch_size, train_name_list, num_fibers_bundles, line_numbers, Track_label_dictionary)

        ## Extract File Names
        for i in range(0, batch_size ):
            fiber_names = train_name_list[line_numbers[i]]
            # print(fiber_names)
            fiber_file_name=fiber_names.strip(' \n')
            a = np.loadtxt(fiber_file_name)
            b=a.flatten()
            batch_X[i,] = b


        return batch_X, batch_Y


    def Create_batchY(self, batch_size, fiber_names_list, num_fibers, line_numbers, Track_lookup_table):

        label_Y = np.full((batch_size, num_fibers ), 0, dtype= 'int')

        for i in range(0, len(line_numbers)):
            fiber_name_i = fiber_names_list[line_numbers[i]]
            fiber_name_i_split = fiber_name_i.split("/")
            batch_Y_row = np.full((1, num_fibers ), 0, dtype='int')
            num_label=(Track_lookup_table[fiber_name_i_split[13]])
            batch_Y_row[0, num_label-1] = 1
            label_Y[i,] = batch_Y_row


        return label_Y
