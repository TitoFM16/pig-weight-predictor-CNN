# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array,load_img

# Procesamiento,manejo y graficación de imagenes
import cv2
import matplotlib.pyplot as plt
import numpy as np

# OS y random
import os, random

# Parametros Batch Size, Epochs ,Image size 
batch_size = 16
epochs = 50
image_size = (64, 64)



# COLOR CNN 
num_skipped = 0


# Directorio de Entrenamiento
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './imagenes/imagenes_red/Train_data',
    validation_split=0.2,
    subset="training",
    seed=15,
    image_size=image_size,
    batch_size=batch_size,
)

# Directorio de Validación
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './imagenes/imagenes_red/Val_data',
    validation_split=0.2,
    subset="validation",
    seed=15,
    image_size=image_size,
    batch_size=batch_size,
)
# Crear Graficas de 10 x 10 para mostrar las imagenes a utilizar en una cuadricula de 3x3
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

# Definir Data Augmentation a las imagenes ( Rotaciones naturales de las imagenes )
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
# Crear Graficas de 10 x 10 para mostrar una imagene aplicada data augmentation en curadricula 3x3
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# Definir inputs del modelo (imagenes de tamaño image_size)
inputs = keras.Input(shape=image_size)

# Aplicar la data augmentation a las imagenes de entrenamiento y validación
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

# Definir tamaños de muestreo para datos de validación y entrenamiento
train_ds = train_ds.prefetch(buffer_size=8)
val_ds = val_ds.prefetch(buffer_size=8)

# Definir el modelo 
def make_model(input_shape, num_classes):
    # Definir inputs de imagenes de input_shape size
    inputs = keras.Input(shape=input_shape)

    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block

    # Normalizar matrices de las imagenes
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(16, 3, padding="same")(x) #antes 64
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(4096, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

# Crear Modelo
model = make_model(input_shape=image_size + (3,), num_classes=2)

#keras.utils.plot_model(model, show_shapes=True)

# Definir path para guardar checkpoint al realizar el callback en el mejor epoch del entrenamiento.
checkpoint_filepath = './Modelos/MODEL_2/'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Compilar Modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=4.5e-4), #original 3.4e-4 - then 4.5e-4
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
# Ajustar Modelo a las imagenes mostradas
model.fit(
    train_ds, epochs=epochs, callbacks=model_checkpoint_callback, validation_data=val_ds,
)



# Path de imagenes de prueba.

#data = "/content/drive/Shareddrives/Proyecto Cerdos/Procesamiento_videos_test_1/Imagenes/imagenesRedNeuronal/prueba malas y buenas"
data = "./prueba_frames/"
prueba = random.choice(os.listdir(data)) #change dir name to whatever
path = data+'/'+prueba

# Cargar imagen y realizar preprocesamiento
img = keras.preprocessing.image.load_img(
    path, target_size=image_size
)
# Convertir a matriz de datos para pasar al modelo
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# Realizar predicción
predictions = model.predict(img_array)

score = predictions[0]

print(predictions)
print(len(predictions))

print(
    "En esta imagen hay un cerdo derecho con una seguridad del %.2f %% y no lo hay con una seguridad del %.2f %%."%(100*(1 - float(score)), 100 * score)
)
# Mostrar Imagen[]
imagen = cv2.imread(path)

#show image
cv2.imshow("Imagen", imagen)