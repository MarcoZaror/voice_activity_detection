## README file 

The solution focuses on detecting voice activity in a bunch of noisy files using two possible architectures; either using a feedforward network or a recurrent neural network:

Assumptions for data:
 - It is expected that the audio files for training, validation and testing are in a folder called audio.
 - It is expected that the labels for training, validation and testing are in a folder called labels.
 - Prefixes NIS and VIT are used for training, EDI for validation and CMU for testing. 

Scripts to train and evaluate each type of model are available in the folder scripts
