import numpy as np
from tensorflow.keras.preprocessing import image


img_size = 128 


model = load_model("cat_dog_model.h5")


test_img = image.load_img("cat.jpg", target_size=(img_size, img_size))
test_img = image.img_to_array(test_img) / 255.0 
test_img = np.expand_dims(test_img, axis=0)  


prediction = model.predict(test_img)[0][0]  
if prediction > 0.5:
    print("It's a DOG! 🐶")
else:
    print("It's a CAT! 🐱")
