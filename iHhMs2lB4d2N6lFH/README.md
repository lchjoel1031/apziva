MonRead:

In this project, I used different CNN models -- VGG16, ResNet50, MobileNet -- to determine whether a image shows flipped or not flipped page from camera. 
The models are trained/refined using F1 score. I used MobileNet as the production model as it provides light-weight models (<40 MB) that can comfortably
fit to storage size of a mobile app. 
A notebook walking through all steps is saved in notebook/, and the source code to run the training is saved in src/train.py, as well as the source code 
to apply the bert model to new dataset in src/app.py
