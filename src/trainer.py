from load_images import load_data
from cnn_model import build_model

train_data, val_data = load_data("src/dataset")


model = build_model()

epochs = 50
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

model.save("cat_dog_model.h5")

