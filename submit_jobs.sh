#$ -S /bin/bash
#$ -o /ifs/loni/faculty/thompson/adni2/scan_1/Vikash_process/Softwares/MATLAB/DeepLearning_Clustering/python_scripts/final_scripts/Training_inverse_tracks_25_points -j y
script_dir=/ifs/loni/faculty/thompson/adni2/scan_1/Vikash_process/Softwares/MATLAB/DeepLearning_Clustering/python_scripts/final_scripts/Training_inverse_tracks_25_points
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/BreakDataIndividualBundles.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/RandomiseFileNames.py
/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/Main_multilayer_perceptron_25points.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/Main_multilayer_perceptron_25points_3_layers.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/MakeNewDataSet.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/RandomiseFileNames_newDataSet.py
# /ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/Main_multilayer_perceptron.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/Main_multilayer_perceptron_3_layers.py
#/ifshome/vgupta/miniconda3/bin/python3.5 ${script_dir}/Main_multilayer_perceptron_1_layer.py
