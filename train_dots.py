# Dependencies : 
#   Keras                2.3.1 
#   tensorflow           2.1.0 

import json;
from matplotlib import pyplot as plt
import numpy as np 
import keras;
import random;
import math;
import os;
import tensorflow as tf;
from keras.layers import Input, Dense, Activation
from PIL import Image

# Load points from JSON
def load_points_json(filename):
    result_x=[];
    result_y=[];
    with open(filename) as f:
        data=json.load(f);
        for p in data:
            result_x.append(p['x']);
            result_y.append(p['y']);
    return {"x":result_x,"y":result_y};


# Save the contents of a numpy array with shape(_,_,3) and values \in [0:1]^2 to RGB image with target_filename 
def save_numpy_rgb(target_filename, npdata):
    rgbArray = np.zeros((npdata.shape[0],npdata.shape[1],3), 'uint8');
    rgbArray[:,:,:]=255.0*npdata[:,:,:];
    rgbImage = (Image.fromarray(rgbArray)).convert('RGB')
    rgbImage.save(target_filename)


    
# Convert the data read into numpy arrays
def setup_train_set(green_points, red_points):
    count=len(green_points['x'])+len(red_points['y']);
    
    points=[];
    for i in range(len(green_points['x'])):
        points.append({'x':green_points['x'][i], 'y': green_points['y'][i], 'v':0})
    for i in range(len(red_points['x'])):
        points.append({'x':red_points['x'][i], 'y': red_points['y'][i], 'v':1})
        
    random.shuffle(points);

    input_data=np.zeros((len(points), 2), 'single');
    output_data=np.zeros((len(points), 2), 'single');
    for i in range(len(points)):
        input_data[i,0]=points[i]['x'];
        input_data[i,1]=points[i]['y'];
        output_data[i,0] = points[i]['v'];
        output_data[i,1] = 1.0-points[i]['v'];
        
    return input_data, output_data;    


# create simple model with inner_layers inner layers, inner_nodes being the number of nodes in each inner layer
# Use binary_crossentropy as loss
def simple_model(inner_nodes, inner_layers):
    input_layer = Input(shape=(2));
    x = input_layer;
    for i in range(inner_layers):
        x = Dense(inner_nodes, activation='relu')(x);
    output_layer = Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=['accuracy'])
    model.summary();
    return model    

# Call back -- Called at end of every epoch
def output_sample(batch, logs):
    global xy_array;
    global model;
    global output_path;
    global o;
    
    if (o % 100)==0:
        result = model.predict(xy_array);
        vis = np.zeros((256, 256, 3), 'single');
        for x in range(256):
            for y in range(256):
                vis[y,x,0:2] = result[256*y+x];
            
        save_numpy_rgb(output_path+"/"+"vis_"+str(o)+".png", vis);
    # increase our epoch counter
    o += 1;    
            

# Load data
green_points=load_points_json("green_points.json");
red_points=load_points_json("red_points.json");
print("I have ", len(green_points['x']), " green points and ", len(red_points['x']), " red points.");

# Setup data (and shuffle)
input_data, output_data = setup_train_set(green_points, red_points);

# Create figure of the data
plt.style.use('seaborn-whitegrid')
plt.plot(green_points['x'], green_points['y'], 'o', color='green');
plt.plot(red_points['x'], red_points['y'], 'o', color='red');
plt.savefig('red_green.png')

# Create input values for all pixels in the output image; That's [0:1]x[0:1]. 
# We'll use xy_array for predicting in output_sample(...)
xy_array = np.zeros((256*256, 2), 'single');
for x in range(256):
    for y in range(256):
        xy_array[256*y+x,0]=x/255.0;
        xy_array[256*y+x,1]=1.0-y/255.0;

# Try meta-parameters
for inner_layers in [1,2,3,4,6,8]:
    for inner_nodes in [2,4,8,16,32,64,128,256,384]:
        # Create output folder
        output_path ="reluout_"+str(inner_layers)+"_"+str(inner_nodes); 
        if not os.path.exists(output_path):
            os.mkdir(output_path);

        o = 0;

        # Model and train
        model = simple_model(inner_nodes, inner_layers);
        history = model.fit(x=input_data, y=output_data, epochs=4000, verbose=True,callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=output_sample)]);

        # Output model after last epoch
        output_sample(None, None)

        # Output loss to file
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(output_path + '/loss.png')
        
