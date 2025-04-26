from load_images import load_data
from cnn_model import build_model

# Load dataset
train_data, val_data = load_data("src/dataset")


# Build CNN model
model = build_model()

# Train model
epochs = 50
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save trained model
model.save("cat_dog_model.h5")
print("Model saved successfully! 🎉")
