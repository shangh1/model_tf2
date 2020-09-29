The goal is to include the data processing into the model graph, so that the data processing can be included into the saved model. 

toy_example_tf2_all_in_saved_model.py: this code includes the data processing (feature transformation) code in the model graph, and the code does not work. Note that the data processing code is inside the function 'toy_tf_util.MecKaminoFeature', which is all written in tf function.

toy_example_tf2.py: this code processes data outside of the model graph, and this code works. But when we do inference, we have to do two-step: first processing data, then letting the model predict the processed data. This two-step way does not satifsy our production's requirements. 
