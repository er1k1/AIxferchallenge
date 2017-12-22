# AIxferchallenge
STL10 Transfer Learning Project

This project uses the unlabeled and labeled images from the STL10 data set aquired from ImageNet. A description is given here https://cs.stanford.edu/~acoates/stl10/ . A Convolutional Neural Network is pre-trained using an autoencoder on the unlabeled images. Then the trained weights are fixed and further fully connected training layers are added which use the labeled training images starting with the pre-trained weights. 
For comparison the entire Convolutional Neural Network was trained solely on the same labeled images with random starting weights for 50 epochs.

The Python code for setting up and running the training excercise can be found in autocnn.py in the root directory. Data loading functions can be found in stl10_input.py in the dataset directory. This is imported by autocnn.py. The trained models are serialised to the output/models directory and if they have already been created and exist in that directory, they are loaded directly instead of retraining. The model histories are also stored in the same place - giving loss curves over the training epochs for the three models (<a href="https://github.com/er1k1/AIxferchallenge/blob/master/autoencloss.png" >autoencoding model</a>, pretrained model and labelled training model)
 
