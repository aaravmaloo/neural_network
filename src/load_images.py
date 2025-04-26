import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_size = 128
batch_size = 32

def load_data(dataset_path="dataset"):

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    return train_data, val_data
