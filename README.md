# Adverasrial-attack-detection-using-manifold-manipulation

README.md
### ABOUT

This repository contains code implementation of the paper "[Adversarial Defense using Targeted Manifold Manipulation]

 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `pytorch==1.8.1`
- `numpy==1.20.3`
- `matplotlib==3.5.2`
- `torchattacks==3.3.0`
- `sklear= 0.22`
- `pandas= 1.5.0`


Our code is tested on `Python 3.8.13`

### How to train a trapdoored mode

To train model from scratch: 

    run: python train_TMM.py

data_name =  change accordingly
batch_size, num_epoch have been set to default, can be changed as per requirements.

### Note :-> 
CIFAR10 trained model has been provided  to reproduce the results



### How to run attack and detection: 

To test detection performance:
1.For Offline Detection - ( only successful attacks are being  tested for detedtion, so TP and FN are measured)

    run : python TMM_O.py 
    
2.For Live Detection -

    run : TMM_L.py 
3.For Advance Live Detection-

    run : TMM_A.py

data_name, attack_name, target mode, and other attack parameters are modifiable.




``
