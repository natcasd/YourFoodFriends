# YourFoodFriends

## Introduction
With a simple picture, YourFoodFriends offers an aid to accurately identify the foods on a plate or tray.  Trained on 101 foods ranging from all cultures and cuisines, our segmentation-classification ensemble seeks to help the visually impaired and tourists know what they are eating.

### Classification
All materials related to the food-classification part of this project can be found in the classification/ folder. Most importantly, subfolder classification/model_runners include the various model architectures that were considered. These files can be run via Python 3 without refactoring, and they will automatically generate checkpoints and logs in the classification/ folder. 

Such models can be further fine-tuned via running classification/food_finetuning.py. This will generate new checkpoints inside classification/finetuned-checkpoints.

Python file classification/evaluate_model.py can be run via Python 3 to evaluate a particular model, though it must be refactored to specify the filepath to a particular checkpoint.

### Segmentation
All materials related to the segmentation part of this project are located in the segmentation/ folder. This primarily includes segmentation/sam_mask_generator.py, which can be run via Python 3 to obtain all masks for a given image. Additionally, several filtering heuristics can be defined in this file.

### Other folders
Folder data/ includes excerpts of the food data upon which our classifer was trained. Please refer to the associated report for links to the full datasets.

Folder fnf/ includes experiments into a food-not-food binary classifier along with associated model architectures, checkpoints, and logs. 

## Running the ensemble
Running Python file classifysegments.py via Python 3 is the primary way to run our ensemble. However, it cannot be run without the model file, which is linked below. The model architecture and training loops are specified in classification/model_runners/food_IRNv2.py. By training that model, one can obtain the checkpoints saved by ModelCheckpoint in classification/checkpoints/. Refactor the file paths in classifysegments.py to load the saved model in question in order to run the ensemble.

This code was optimized to work on Oscar CCV and uses mixed precision training, which is optimized for more recent NVIDIA GPUs. 


## Additional Information
Our poster is titled "Poster - YourFoodFriends.pdf"

This is a link to our GitHub if that makes for easier viewing:
https://github.com/jrbyers/YourFoodFriends

This is a link to the fine-tuned Inception ResNet V2 model used for food classification (remember to properly refactor classifysegments.py to properly run the ensemble): https://drive.google.com/drive/folders/1WdiAyniCfnin1f2wUCTttdFEQh71PD4Z?usp=share_link. 

This notebook was helpful for setting up our initial classifier training loop: https://dev.mrdbourke.com/tensorflow-deep-learning/07_food_vision_milestone_project_1/

## Acknowledgements
This ensemble was developed by Winston Li, John Ryan Byers, and Nathan DePiero. We thank the Brown CSCI1430 Team and particularly TA Anh Duong and Professor James Tompkin for their endless generosity and guidance.


