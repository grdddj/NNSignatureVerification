# Verification signature using Neural Network
This program is trying to verificate offline signatures with usage of two methods using neural network.
## Instalation
Program is installed by downloading the code (or cloning) and installng requirments.txt. Create data folder and download there a Cedar dataset. Then running python main.py. 
## Files in program
### Data
In data folder download CEDAR dataset.
### main.py
Running main.py set up everything, starts the program and let user decide the parameters.
### model.py
Contains model architecture.
### loader.py
Includes dataloaders, and augmentation functions
### functions.py
Contains supportive functions for models and loader such as callbacks, additional feature extractions
### tester.py
Contains functions for testing trained models.
