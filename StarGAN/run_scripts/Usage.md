# Preprocess
First the data has to be preprocessedm via the preprocessing script, this must be done for all speakers included in the conversion later on

# Train
The model will then be trained via the train script and all the preprocessed from the train folder given will be included. As dimensions must match using an already trained model, reguires this to have been trained on the same speakers.

# Convert
Using the convert script, conversion can be done between the voices from the model, but with the test set created via preprocessing.
