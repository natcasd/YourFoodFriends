# YourFoodFriends
Structure:
classifysegments.py joins the classifier and the segmenter model. In classification/model_runners you can see the different models we iterated through. In segmentation/sam_mask_generator you can see how we used SAM.

classifysegments.py is the main file. However, it can not be run without the model file, which is too big to include in github. The model architecture and train loop are specified in classification/model_runners/food_IRNv2.py, so by training that model, using model.save() in tensorflow, and refactoring file paths in classifysegments.py to load that model, it can be run. This code was optimized to work on Oscar computing cluster.