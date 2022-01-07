# program that loads a tensorflow image clasification model and test it with a loaded video analysing every video frame
import cv2
import numpy as np
import random
import os
import re
import tensorflow as tf
import argparse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array,load_img

# construct argument parser to get video path
    #construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the images directory")

args = vars(ap.parse_args())
# Get the image path
imagePath = args["path"]

image_size = (64, 64)

# Definir Data Augmentation a las imagenes ( Rotaciones naturales de las imagenes )
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

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

# Loads the weights
# Create a basic model instance
# Crear Modelo
model = make_model(input_shape=image_size + (3,), num_classes=2)
checkpoint_filepath = './Modelos/MODEL_2/'
model.load_weights(checkpoint_filepath)

# video path
#video = "C:/Users/juant/Workspace/cerdos/videos/C1/Cerdos_mod1_20210823_0851.avi"
video = imagePath
# if video has "C1" pigs are male else female
f = re.findall('C\d{1}',video)
if f[0] =='C1':
    pigs = 'C1'

elif f[0] =='C2':
    pigs = 'C2'

#read video with opencv
cap = cv2.VideoCapture(video)
# for every frame in video binarize image, find contours and draw ellipse
frame_video = -1
while(cap.isOpened()):
    frame_video+=1
    print(frame_video)
    ret, frame = cap.read()
    if ret == True:
        #binarize image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply threshold
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # use morphology to get rid of noise
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        #find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            if  area > 3500 and area < 20000:
                #get the bounding rectangle
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #crop the bounding box
                x1 = box[0][0]; x2 = box[1][0]; x3 = box[2][0]; x4 = box[3][0]
                y1 = box[0][1]; y2 = box[1][1]; y3 = box[2][1]; y4 = box[3][1]

                top_left_x  = min([x1,x2,x3,x4])
                top_left_y  = min([y1,y2,y3,y4])
                bot_right_x = max([x1,x2,x3,x4])
                bot_right_y = max([y1,y2,y3,y4])
                
                crop = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
              
                #get cropped image size
                crop_height, crop_width = crop.shape[:2]

                #show cropped image and ask pigs weight to user
                try:
                    
                    # Cargar imagen y realizar preprocesamiento
                    # Convertir a matriz de datos para pasar al modelo
                    img_array = keras.preprocessing.image.img_to_array(cv2.resize(crop, (64,64)))
                    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                    # Realizar predicción
                    predictions = model.predict(img_array)
                    score = predictions[0]
                    # get bounding ellipse
                    ellipse = cv2.fitEllipse(cnt)
                    (xc,yc),(d1,d2),angle = ellipse
                    eje_mayor = max(d1,d2)
                    eje_menor = min(d1,d2)
                    area = np.pi*eje_mayor*eje_menor
                    relacion_ejes = eje_menor/eje_mayor
                    print('area',area,'relacion ejes',relacion_ejes,'eje mayor',eje_mayor,'Eje menor',eje_menor)
                    if area < 70000 and area > 25000 and relacion_ejes >0.2 and relacion_ejes <0.33: # and eje_mayor<400 and eje_menor<80                
                        
                        relacion_ejes = eje_menor/eje_mayor                        
                        if pigs == "C1":
                            equation = 0.000501*area-3.143853
                        else:
                            equation = 0.000559*area-4.158189
                        print('peso',equation)
                        # draw ellipse
                        cv2.ellipse(frame,ellipse,(0,255,0),2)
                        # show ellipse area in elipse bounding and relacion_ejes

                    
                    cv2.imshow('cropped', crop)
                    
                                                                      
                    if 100*(1 - float(score)) >90:
                  
                        cv2.putText(frame,"Cerdo %.2f"%(100*(1 - float(score))), (int(xc),int(yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame,f"Peso: {equation}", (int(xc),int(yc)+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.imshow('cropped', crop)
                        
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    print(
                        "En esta imagen hay un cerdo derecho con una seguridad del %.2f %% y no lo hay con una seguridad del %.2f %%."%(100*(1 - float(score)), 100 * score)
                    )
                except Exception as e:
                    print(e)
                    
                    continue
      
        # show image
        cv2.imshow('frame', frame)
        
        # press q to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()



# Re-evaluate the model


# Mostrar Imagen[]




