# Visualizing 2D fitting with Keras

I saw the following slide in the `MIT 6.S191: Convolutional Neural Networks` lesson, and wondered how fitting would look like. 

![slide](slide.jpg "Slide").
	 
Since this is actually a very simple 2D example, I decided to see how neural nets with different depth and "width" behaves while fitting -- Just for fun. So I decided to try to using Tensorflow. First I created the point sets `green_points.json` and `red_points.json` with a small HTML utitlity where I recorded points clicked.

![points](red_green.png "Slide").

Then I created the small script `train_dots.py`, and trained a small model of the form:

	def simple_model(inner_nodes, inner_layers):
	    input_layer = Input(shape=(2));
	    x = input_layer;
	    for i in range(inner_layers):
	        x = Dense(inner_nodes, activation='sigmoid')(x);
	    output_layer = Dense(2, activation='sigmoid')(x)
	    model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])
	    model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=['accuracy'])
	    model.summary();
	    return model    

This will create a simple model consisting of a number of inner layers each with inner_nodes.

I then tried with:
	inner_layers in [1,2,3,4,6,8]
    inner_nodes in [2,4,8,16,32,64,128,256,384]

For each combination of above parameters I trained the model for 4000 epochs. I then create an image where each pixel is predicted for every 100th epoch (i.e. the image represents all points in the plane). I animate the images to create an animation of the fitting process. The results are in the table below (Final is the final image).
It looks like we need fewer nodes the more layers we have to get a decent looking fit. So 2 layers, 128 nodes, 3 layers 64 nodes, 4 layers 32 nodes all look like good fits. More layers doesn't seem to converge and a single layer doesn't seem sufficient at all. Adding more nodes with 2-4 layers, generally seem to cause faster conversion.

So in conclusion, it is indeed possible to fit a figure around these points, with a few layers and nodes, and about 2000 epochs. 


| Inner layers | nodes | Animation (sigmoid) |  Final (sigmoid) | Loss (sigmoid)   | Animation (relu) |  Final (relu) | Loss (relu)   | 
|--------------|-------|--------|--------|--------|--------|--------|--------|
| 1            |   2   | ![a](out_1_2/out_1_2.gif).     | ![a](out_1_2/vis_4000.png).   | ![a](out_1_2/loss.png).   | ![a](reluout_1_2/reluout_1_2.gif).     | ![a](reluout_1_2/vis_4000.png).   | ![a](reluout_1_2/loss.png).   | 
| 1            |   4   | ![a](out_1_4/out_1_4.gif).     | ![a](out_1_4/vis_4000.png).   | ![a](out_1_4/loss.png).   | ![a](reluout_1_4/reluout_1_4.gif).     | ![a](reluout_1_4/vis_4000.png).   | ![a](reluout_1_4/loss.png).   | 
| 1            |   8   | ![a](out_1_8/out_1_8.gif).     | ![a](out_1_8/vis_4000.png).   | ![a](out_1_8/loss.png).   | ![a](reluout_1_8/reluout_1_8.gif).     | ![a](reluout_1_8/vis_4000.png).   | ![a](reluout_1_8/loss.png).   | 
| 1            |   16  | ![a](out_1_16/out_1_16.gif).   | ![a](out_1_16/vis_4000.png).  | ![a](out_1_16/loss.png).  | ![a](reluout_1_16/reluout_1_16.gif).   | ![a](reluout_1_16/vis_4000.png).  | ![a](reluout_1_16/loss.png).  | 
| 1            |   32  | ![a](out_1_32/out_1_32.gif).   | ![a](out_1_32/vis_4000.png).  | ![a](out_1_32/loss.png).  | ![a](reluout_1_32/reluout_1_32.gif).   | ![a](reluout_1_32/vis_4000.png).  | ![a](reluout_1_32/loss.png).  | 
| 1            |   64  | ![a](out_1_64/out_1_64.gif).   | ![a](out_1_64/vis_4000.png).  | ![a](out_1_64/loss.png).  | ![a](reluout_1_64/reluout_1_64.gif).   | ![a](reluout_1_64/vis_4000.png).  | ![a](reluout_1_64/loss.png).  | 
| 1            |   128 | ![a](out_1_128/out_1_128.gif). | ![a](out_1_128/vis_4000.png). | ![a](out_1_128/loss.png). | ![a](reluout_1_128/reluout_1_128.gif). | ![a](reluout_1_128/vis_4000.png). | ![a](reluout_1_128/loss.png). | 
| 1            |   256 | ![a](out_1_256/out_1_256.gif). | ![a](out_1_256/vis_4000.png). | ![a](out_1_256/loss.png). | ![a](reluout_1_256/reluout_1_256.gif). | ![a](reluout_1_256/vis_4000.png). | ![a](reluout_1_256/loss.png). | 
| 1            |   384 | ![a](out_1_384/out_1_384.gif). | ![a](out_1_384/vis_4000.png). | ![a](out_1_384/loss.png). | ![a](reluout_1_384/reluout_1_384.gif). | ![a](reluout_1_384/vis_4000.png). | ![a](reluout_1_384/loss.png). | 
| 2            |   2   | ![a](out_2_2/out_2_2.gif).     | ![a](out_2_2/vis_3999.png).   | ![a](out_2_2/loss.png).   | ![a](reluout_2_2/reluout_2_2.gif).     | ![a](reluout_2_2/vis_4000.png).   | ![a](reluout_2_2/loss.png).   | 
| 2            |   4   | ![a](out_2_4/out_2_4.gif).     | ![a](out_2_4/vis_3999.png).   | ![a](out_2_4/loss.png).   | ![a](reluout_2_4/reluout_2_4.gif).     | ![a](reluout_2_4/vis_4000.png).   | ![a](reluout_2_4/loss.png).   | 
| 2            |   8   | ![a](out_2_8/out_2_8.gif).     | ![a](out_2_8/vis_3999.png).   | ![a](out_2_8/loss.png).   | ![a](reluout_2_8/reluout_2_8.gif).     | ![a](reluout_2_8/vis_4000.png).   | ![a](reluout_2_8/loss.png).   | 
| 2            |   16  | ![a](out_2_16/out_2_16.gif).   | ![a](out_2_16/vis_3999.png).  | ![a](out_2_16/loss.png).  | ![a](reluout_2_16/reluout_2_16.gif).   | ![a](reluout_2_16/vis_4000.png).  | ![a](reluout_2_16/loss.png).  | 
| 2            |   32  | ![a](out_2_32/out_2_32.gif).   | ![a](out_2_32/vis_3999.png).  | ![a](out_2_32/loss.png).  | ![a](reluout_2_32/reluout_2_32.gif).   | ![a](reluout_2_32/vis_4000.png).  | ![a](reluout_2_32/loss.png).  | 
| 2            |   64  | ![a](out_2_64/out_2_64.gif).   | ![a](out_2_64/vis_3999.png).  | ![a](out_2_64/loss.png).  | ![a](reluout_2_64/reluout_2_64.gif).   | ![a](reluout_2_64/vis_4000.png).  | ![a](reluout_2_64/loss.png).  | 
| 2            |   128 | ![a](out_2_128/out_2_128.gif). | ![a](out_2_128/vis_3999.png). | ![a](out_2_128/loss.png). | ![a](reluout_2_128/reluout_2_128.gif). | ![a](reluout_2_128/vis_4000.png). | ![a](reluout_2_128/loss.png). | 
| 2            |   256 | ![a](out_2_256/out_2_256.gif). | ![a](out_2_256/vis_3999.png). | ![a](out_2_256/loss.png). | ![a](reluout_2_256/reluout_2_256.gif). | ![a](reluout_2_256/vis_4000.png). | ![a](reluout_2_256/loss.png). | 
| 2            |   384 | ![a](out_2_384/out_2_384.gif). | ![a](out_2_384/vis_3999.png). | ![a](out_2_384/loss.png). | ![a](reluout_2_384/reluout_2_384.gif). | ![a](reluout_2_384/vis_4000.png). | ![a](reluout_2_384/loss.png). | 
| 3            |   2   | ![a](out_3_2/out_3_2.gif).     | ![a](out_3_2/vis_3999.png).   | ![a](out_3_2/loss.png).   | ![a](reluout_3_2/reluout_3_2.gif).     | ![a](reluout_3_2/vis_4000.png).   | ![a](reluout_3_2/loss.png).   | 
| 3            |   4   | ![a](out_3_4/out_3_4.gif).     | ![a](out_3_4/vis_3999.png).   | ![a](out_3_4/loss.png).   | ![a](reluout_3_4/reluout_3_4.gif).     | ![a](reluout_3_4/vis_4000.png).   | ![a](reluout_3_4/loss.png).   | 
| 3            |   8   | ![a](out_3_8/out_3_8.gif).     | ![a](out_3_8/vis_3999.png).   | ![a](out_3_8/loss.png).   | ![a](reluout_3_8/reluout_3_8.gif).     | ![a](reluout_3_8/vis_4000.png).   | ![a](reluout_3_8/loss.png).   | 
| 3            |   16  | ![a](out_3_16/out_3_16.gif).   | ![a](out_3_16/vis_3999.png).  | ![a](out_3_16/loss.png).  | ![a](reluout_3_16/reluout_3_16.gif).   | ![a](reluout_3_16/vis_4000.png).  | ![a](reluout_3_16/loss.png).  | 
| 3            |   32  | ![a](out_3_32/out_3_32.gif).   | ![a](out_3_32/vis_3999.png).  | ![a](out_3_32/loss.png).  | ![a](reluout_3_32/reluout_3_32.gif).   | ![a](reluout_3_32/vis_4000.png).  | ![a](reluout_3_32/loss.png).  | 
| 3            |   64  | ![a](out_3_64/out_3_64.gif).   | ![a](out_3_64/vis_3999.png).  | ![a](out_3_64/loss.png).  | ![a](reluout_3_64/reluout_3_64.gif).   | ![a](reluout_3_64/vis_4000.png).  | ![a](reluout_3_64/loss.png).  | 
| 3            |   128 | ![a](out_3_128/out_3_128.gif). | ![a](out_3_128/vis_3999.png). | ![a](out_3_128/loss.png). | ![a](reluout_3_128/reluout_3_128.gif). | ![a](reluout_3_128/vis_4000.png). | ![a](reluout_3_128/loss.png). | 
| 3            |   256 | ![a](out_3_256/out_3_256.gif). | ![a](out_3_256/vis_3999.png). | ![a](out_3_256/loss.png). | ![a](reluout_3_256/reluout_3_256.gif). | ![a](reluout_3_256/vis_4000.png). | ![a](reluout_3_256/loss.png). | 
| 3            |   384 | ![a](out_3_384/out_3_384.gif). | ![a](out_3_384/vis_3999.png). | ![a](out_3_384/loss.png). | ![a](reluout_3_384/reluout_3_384.gif). | ![a](reluout_3_384/vis_4000.png). | ![a](reluout_3_384/loss.png). | 
| 4            |   2   | ![a](out_4_2/out_4_2.gif).     | ![a](out_4_2/vis_3999.png).   | ![a](out_4_2/loss.png).   | ![a](reluout_4_2/reluout_4_2.gif).     | ![a](reluout_4_2/vis_4000.png).   | ![a](reluout_4_2/loss.png).   | 
| 4            |   4   | ![a](out_4_4/out_4_4.gif).     | ![a](out_4_4/vis_3999.png).   | ![a](out_4_4/loss.png).   | ![a](reluout_4_4/reluout_4_4.gif).     | ![a](reluout_4_4/vis_4000.png).   | ![a](reluout_4_4/loss.png).   | 
| 4            |   8   | ![a](out_4_8/out_4_8.gif).     | ![a](out_4_8/vis_3999.png).   | ![a](out_4_8/loss.png).   | ![a](reluout_4_8/reluout_4_8.gif).     | ![a](reluout_4_8/vis_4000.png).   | ![a](reluout_4_8/loss.png).   | 
| 4            |   16  | ![a](out_4_16/out_4_16.gif).   | ![a](out_4_16/vis_3999.png).  | ![a](out_4_16/loss.png).  | ![a](reluout_4_16/reluout_4_16.gif).   | ![a](reluout_4_16/vis_4000.png).  | ![a](reluout_4_16/loss.png).  | 
| 4            |   32  | ![a](out_4_32/out_4_32.gif).   | ![a](out_4_32/vis_3999.png).  | ![a](out_4_32/loss.png).  | ![a](reluout_4_32/reluout_4_32.gif).   | ![a](reluout_4_32/vis_4000.png).  | ![a](reluout_4_32/loss.png).  | 
| 4            |   64  | ![a](out_4_64/out_4_64.gif).   | ![a](out_4_64/vis_3999.png).  | ![a](out_4_64/loss.png).  | ![a](reluout_4_64/reluout_4_64.gif).   | ![a](reluout_4_64/vis_4000.png).  | ![a](reluout_4_64/loss.png).  | 
| 4            |   128 | ![a](out_4_128/out_4_128.gif). | ![a](out_4_128/vis_3999.png). | ![a](out_4_128/loss.png). | ![a](reluout_4_128/reluout_4_128.gif). | ![a](reluout_4_128/vis_4000.png). | ![a](reluout_4_128/loss.png). | 
| 4            |   256 | ![a](out_4_256/out_4_256.gif). | ![a](out_4_256/vis_3999.png). | ![a](out_4_256/loss.png). | ![a](reluout_4_256/reluout_4_256.gif). | ![a](reluout_4_256/vis_4000.png). | ![a](reluout_4_256/loss.png). | 
| 4            |   384 | ![a](out_4_384/out_4_384.gif). | ![a](out_4_384/vis_3999.png). | ![a](out_4_384/loss.png). | ![a](reluout_4_384/reluout_4_384.gif). | ![a](reluout_4_384/vis_4000.png). | ![a](reluout_4_384/loss.png). | 
| 6            |   2   | ![a](out_6_2/out_6_2.gif).     | ![a](out_6_2/vis_3999.png).   | ![a](out_6_2/loss.png).   | ![a](reluout_6_2/reluout_6_2.gif).     | ![a](reluout_6_2/vis_4000.png).   | ![a](reluout_6_2/loss.png).   | 
| 6            |   4   | ![a](out_6_4/out_6_4.gif).     | ![a](out_6_4/vis_3999.png).   | ![a](out_6_4/loss.png).   | ![a](reluout_6_4/reluout_6_4.gif).     | ![a](reluout_6_4/vis_4000.png).   | ![a](reluout_6_4/loss.png).   | 
| 6            |   8   | ![a](out_6_8/out_6_8.gif).     | ![a](out_6_8/vis_3999.png).   | ![a](out_6_8/loss.png).   | ![a](reluout_6_8/reluout_6_8.gif).     | ![a](reluout_6_8/vis_4000.png).   | ![a](reluout_6_8/loss.png).   | 
| 6            |   16  | ![a](out_6_16/out_6_16.gif).   | ![a](out_6_16/vis_3999.png).  | ![a](out_6_16/loss.png).  | ![a](reluout_6_16/reluout_6_16.gif).   | ![a](reluout_6_16/vis_4000.png).  | ![a](reluout_6_16/loss.png).  | 
| 6            |   32  | ![a](out_6_32/out_6_32.gif).   | ![a](out_6_32/vis_3999.png).  | ![a](out_6_32/loss.png).  | ![a](reluout_6_32/reluout_6_32.gif).   | ![a](reluout_6_32/vis_4000.png).  | ![a](reluout_6_32/loss.png).  | 
| 6            |   64  | ![a](out_6_64/out_6_64.gif).   | ![a](out_6_64/vis_3999.png).  | ![a](out_6_64/loss.png).  | ![a](reluout_6_64/reluout_6_64.gif).   | ![a](reluout_6_64/vis_4000.png).  | ![a](reluout_6_64/loss.png).  | 
| 6            |   128 | ![a](out_6_128/out_6_128.gif). | ![a](out_6_128/vis_3999.png). | ![a](out_6_128/loss.png). | ![a](reluout_6_128/reluout_6_128.gif). | ![a](reluout_6_128/vis_4000.png). | ![a](reluout_6_128/loss.png). | 
| 6            |   256 | ![a](out_6_256/out_6_256.gif). | ![a](out_6_256/vis_3999.png). | ![a](out_6_256/loss.png). | ![a](reluout_6_256/reluout_6_256.gif). | ![a](reluout_6_256/vis_4000.png). | ![a](reluout_6_256/loss.png). | 
| 6            |   384 | ![a](out_6_384/out_6_384.gif). | ![a](out_6_384/vis_3999.png). | ![a](out_6_384/loss.png). | ![a](reluout_6_384/reluout_6_384.gif). | ![a](reluout_6_384/vis_4000.png). | ![a](reluout_6_384/loss.png). | 
| 8            |   2   | ![a](out_8_2/out_8_2.gif).     | ![a](out_8_2/vis_3999.png).   | ![a](out_8_2/loss.png).   | ![a](reluout_8_2/reluout_8_2.gif).     | ![a](reluout_8_2/vis_4000.png).   | ![a](reluout_8_2/loss.png).   | 
| 8            |   4   | ![a](out_8_4/out_8_4.gif).     | ![a](out_8_4/vis_3999.png).   | ![a](out_8_4/loss.png).   | ![a](reluout_8_4/reluout_8_4.gif).     | ![a](reluout_8_4/vis_4000.png).   | ![a](reluout_8_4/loss.png).   | 
| 8            |   8   | ![a](out_8_8/out_8_8.gif).     | ![a](out_8_8/vis_3999.png).   | ![a](out_8_8/loss.png).   | ![a](reluout_8_8/reluout_8_8.gif).     | ![a](reluout_8_8/vis_4000.png).   | ![a](reluout_8_8/loss.png).   | 
| 8            |   16  | ![a](out_8_16/out_8_16.gif).   | ![a](out_8_16/vis_3999.png).  | ![a](out_8_16/loss.png).  | ![a](reluout_8_16/reluout_8_16.gif).   | ![a](reluout_8_16/vis_4000.png).  | ![a](reluout_8_16/loss.png).  | 
| 8            |   32  | ![a](out_8_32/out_8_32.gif).   | ![a](out_8_32/vis_3999.png).  | ![a](out_8_32/loss.png).  | ![a](reluout_8_32/reluout_8_32.gif).   | ![a](reluout_8_32/vis_4000.png).  | ![a](reluout_8_32/loss.png).  | 
| 8            |   64  | ![a](out_8_64/out_8_64.gif).   | ![a](out_8_64/vis_3999.png).  | ![a](out_8_64/loss.png).  | ![a](reluout_8_64/reluout_8_64.gif).   | ![a](reluout_8_64/vis_4000.png).  | ![a](reluout_8_64/loss.png).  | 
| 8            |   128 | ![a](out_8_128/out_8_128.gif). | ![a](out_8_128/vis_3999.png). | ![a](out_8_128/loss.png). | ![a](reluout_8_128/reluout_8_128.gif). | ![a](reluout_8_128/vis_4000.png). | ![a](reluout_8_128/loss.png). | 
| 8            |   256 | ![a](out_8_256/out_8_256.gif). | ![a](out_8_256/vis_3999.png). | ![a](out_8_256/loss.png). | ![a](reluout_8_256/reluout_8_256.gif). | ![a](reluout_8_256/vis_4000.png). | ![a](reluout_8_256/loss.png). | 
| 8            |   384 | ![a](out_8_384/out_8_384.gif). | ![a](out_8_384/vis_3999.png). | ![a](out_8_384/loss.png). | ![a](reluout_8_384/reluout_8_384.gif). | ![a](reluout_8_384/vis_4000.png). | ![a](reluout_8_384/loss.png). | 
