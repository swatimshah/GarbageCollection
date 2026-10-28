These are the model creation and testing files. They load the image data "lazily". They create a model to classify the images into "paper", "plastic" or "leaves". 
At the very beginning of the model creation process, the models at each epoch are stored in the file system. And each of these models is evaluated against each 
testing dataset. The best model is chosen based on average accuracy across all the test datasets. The "inc" files operate on "incremental" datasets. It loads the 
best model and start to train the model from there, with the new set of data with limited iterations and a modified learning rate. Once the incremental model gets
trained, it also gets evaluated for its accuracy with the test datasets. As on today, I could reach maximum incremental accuracy of average 72%. 

