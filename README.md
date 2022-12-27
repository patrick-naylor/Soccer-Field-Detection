# Soccer Field Detection

As a preprocessing step for player and ball tracking models being able to isolate the playing surface is essential.
In order to to this I trained a convolutional neural network on images from the Islamic Azad University Football Dataset. The model is trained on football/soccer images but could easily be adapted to other rectangular field sports. The training data was created by taking images from the database and drawing the bounds of the playing surface using the python script `image_app.py`. The model is trained using `model.ipynb`.

To use the model users can use the field_mask function in `soccerdetect.py` to mask an image contained in a numpy array. The module is designed to be used with cv2. To use this function users will have to download the trained model from [my Kaggle page](https://www.kaggle.com/datasets/patricknaylor/soccer-playing-surface-masks).

Thanks and enjoy!
