## README file 

The solution consists of 5 files:

Assumptions for data:
 - It is expected that the audio files for training, validation and testing are in a folder called audio.
 - It is expected that the labels for training, validation and testing are in a folder called labels.
 - It is expected that the only files in the folders audio and labels, are the files wanted to use.
 - Prefixes NIS and VIT are used for training, EDI for validation and CMU for testing. 

### Files

File: train.py  
Objective: Train a feed-forward neural network or a recurrent neural network  
Input:   
 - Path to folder of audio files  
 - Path to folder of labels  
 - A binary value (1 or 0) to indicate which kind of model to train  
 
Output:  
 - The trained model is stored in the same directory  
 
Command to run: python train.py <folder audio> <folder labels> <0 (ffnn) or 1 (rnn)>  
Example: python train.py audio/ labels/ 0   
* It is assumed that only the files selected for training and validation are in their respective folders  


File: evaluate.py  
Objective: Evaluate an already trained feed-forward neural network or a recurrent neural network  
Input:  
 - Path to folder of audio files  
 - Path to folder of labels  
 - A binary value (1 or 0) to indicate which kind of model will be imported  
 - Name of the model  
 
Command to run: python evaluate.py <folder audio> <folder labels> <0 (ffnn) or 1 (rnn)> <name model>  
Example: python evaluate.py audio/ labels/ 0 model-7  
* It is assumed that only the files selected for validation and testing are in their respective folders  

File: Data.py  
Objective: Provide functions to load, preprocess and enhance data  

File: model.py  
Objective: Provide the definition of two models; a feed-forward neural network and a recurrent neural network  

File: train_eval_func.py  
Objective: Provide functions to evaluate a particular trained model  
