import numpy as np
import  random
import tensorflow as tf

class GeneratePredictionBatch():
    _placeHolder = 0

    def __init__(self):
        _tempPlaceHolder = 0

    def CreatePredictionBatch(self, validation_file_list, start_idx, end_idx, num_points):
        
        batch_size = end_idx - start_idx
        print ('Start Idx: ', start_idx,  'End Idx: ', end_idx, 'Batch_size: ', batch_size)
        batch_X = np.full((batch_size, 3*num_points), 1.0, dtype='float')
        count=0
        for i in range(start_idx, end_idx):
            fiber_name = validation_file_list[i]
            fiber_name_file = fiber_name.strip(' \n')
            a = np.loadtxt(fiber_name_file)
            b = a.flatten()
            batch_X[count, ] = b
            count = count + 1


        return batch_X
