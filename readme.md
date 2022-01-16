# Visualizing 2D fitting with Keras

I saw the following slide in the `MIT 6.S191: Convolutional Neural Networks` lesson, and wondered how fitting would look like. 

![slide](slide.jpg "Slide").
	 
Since this is actually a very simple 2D example, I decided to try with Tensorflow. First I created the point sets `green_points.json` and `red_points.json` with a small HTML utitlity where I recorded points clicked.

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
	inner_layers in [2,3,4,6,8]
    inner_nodes in [2,4,8,16,32,64,128,256,384]

For each combination of above parameters I trained the model for 4000 epochs, and saved the resulting image for every 100th epoch to create an animation of the fit. The results looks like this:

| Inner layers | nodes | Animation     |  Final | Loss   |
|--------------|-------|--------|--------|--------|
| 2            |   2   | ![a](out_2_2/out_2_2.gif). | ![a](out_2_2/vis_3999.png). | ![a](out_2_2/loss.png). | 
| 2            |   4   | ![a](out_2_4/out_2_4.gif). | ![a](out_2_4/vis_3999.png). | ![a](out_2_4/loss.png). | 
| 2            |   8   | ![a](out_2_8/out_2_8.gif). | ![a](out_2_8/vis_3999.png). | ![a](out_2_8/loss.png). | 
| 2            |   16   | ![a](out_2_16/out_2_16.gif). | ![a](out_2_16/vis_3999.png). | ![a](out_2_16/loss.png). | 
| 2            |   32   | ![a](out_2_32/out_2_32.gif). | ![a](out_2_32/vis_3999.png). | ![a](out_2_32/loss.png). | 
| 2            |   64   | ![a](out_2_64/out_2_64.gif). | ![a](out_2_64/vis_3999.png). | ![a](out_2_64/loss.png). | 
| 2            |   128   | ![a](out_2_128/out_2_128.gif). | ![a](out_2_128/vis_3999.png). | ![a](out_2_128/loss.png). | 
| 2            |   256   | ![a](out_2_256/out_2_256.gif). | ![a](out_2_256/vis_3999.png). | ![a](out_2_256/loss.png). | 
| 2            |   384   | ![a](out_2_384/out_2_384.gif). | ![a](out_2_384/vis_3999.png). | ![a](out_2_384/loss.png). | 
|--------------|-------|--------|--------|--------|
| 3            |   2   | ![a](out_3_2/out_3_2.gif). | ![a](out_3_2/vis_3999.png). | ![a](out_3_2/loss.png). | 
| 3            |   4   | ![a](out_3_4/out_3_4.gif). | ![a](out_3_4/vis_3999.png). | ![a](out_3_4/loss.png). | 
| 3            |   8   | ![a](out_3_8/out_3_8.gif). | ![a](out_3_8/vis_3999.png). | ![a](out_3_8/loss.png). | 
| 3            |   16   | ![a](out_3_16/out_3_16.gif). | ![a](out_3_16/vis_3999.png). | ![a](out_3_16/loss.png). | 
| 3            |   32   | ![a](out_3_32/out_3_32.gif). | ![a](out_3_32/vis_3999.png). | ![a](out_3_32/loss.png). | 
| 3            |   64   | ![a](out_3_64/out_3_64.gif). | ![a](out_3_64/vis_3999.png). | ![a](out_3_64/loss.png). | 
| 3            |   128   | ![a](out_3_128/out_3_128.gif). | ![a](out_3_128/vis_3999.png). | ![a](out_3_128/loss.png). | 
| 3            |   256   | ![a](out_3_256/out_3_256.gif). | ![a](out_3_256/vis_3999.png). | ![a](out_3_256/loss.png). | 
| 3            |   384   | ![a](out_3_384/out_3_384.gif). | ![a](out_3_384/vis_3999.png). | ![a](out_3_384/loss.png). | 
|--------------|-------|--------|--------|--------|
| 4            |   2   | ![a](out_4_2/out_4_2.gif). | ![a](out_4_2/vis_3999.png). | ![a](out_4_2/loss.png). | 
| 4            |   4   | ![a](out_4_4/out_4_4.gif). | ![a](out_4_4/vis_3999.png). | ![a](out_4_4/loss.png). | 
| 4            |   8   | ![a](out_4_8/out_4_8.gif). | ![a](out_4_8/vis_3999.png). | ![a](out_4_8/loss.png). | 
| 4            |   16   | ![a](out_4_16/out_4_16.gif). | ![a](out_4_16/vis_3999.png). | ![a](out_4_16/loss.png). | 
| 4            |   32   | ![a](out_4_32/out_4_32.gif). | ![a](out_4_32/vis_3999.png). | ![a](out_4_32/loss.png). | 
| 4            |   64   | ![a](out_4_64/out_4_64.gif). | ![a](out_4_64/vis_3999.png). | ![a](out_4_64/loss.png). | 
| 4            |   128   | ![a](out_4_128/out_4_128.gif). | ![a](out_4_128/vis_3999.png). | ![a](out_4_128/loss.png). | 
| 4            |   256   | ![a](out_4_256/out_4_256.gif). | ![a](out_4_256/vis_3999.png). | ![a](out_4_256/loss.png). | 
| 4            |   384   | ![a](out_4_384/out_4_384.gif). | ![a](out_4_384/vis_3999.png). | ![a](out_4_384/loss.png). | 
|--------------|-------|--------|--------|--------|
| 6            |   2   | ![a](out_6_2/out_6_2.gif). | ![a](out_6_2/vis_3999.png). | ![a](out_6_2/loss.png). | 
| 6            |   4   | ![a](out_6_4/out_6_4.gif). | ![a](out_6_4/vis_3999.png). | ![a](out_6_4/loss.png). | 
| 6            |   8   | ![a](out_6_8/out_6_8.gif). | ![a](out_6_8/vis_3999.png). | ![a](out_6_8/loss.png). | 
| 6            |   16   | ![a](out_6_16/out_6_16.gif). | ![a](out_6_16/vis_3999.png). | ![a](out_6_16/loss.png). | 
| 6            |   32   | ![a](out_6_32/out_6_32.gif). | ![a](out_6_32/vis_3999.png). | ![a](out_6_32/loss.png). | 
| 6            |   64   | ![a](out_6_64/out_6_64.gif). | ![a](out_6_64/vis_3999.png). | ![a](out_6_64/loss.png). | 
| 6            |   128   | ![a](out_6_128/out_6_128.gif). | ![a](out_6_128/vis_3999.png). | ![a](out_6_128/loss.png). | 
| 6            |   256   | ![a](out_6_256/out_6_256.gif). | ![a](out_6_256/vis_3999.png). | ![a](out_6_256/loss.png). | 
| 6            |   384   | ![a](out_6_384/out_6_384.gif). | ![a](out_6_384/vis_3999.png). | ![a](out_6_384/loss.png). | 
|--------------|-------|--------|--------|--------|
| 8            |   2   | ![a](out_8_2/out_8_2.gif). | ![a](out_8_2/vis_3999.png). | ![a](out_8_2/loss.png). | 
| 8            |   4   | ![a](out_8_4/out_8_4.gif). | ![a](out_8_4/vis_3999.png). | ![a](out_8_4/loss.png). | 
| 8            |   8   | ![a](out_8_8/out_8_8.gif). | ![a](out_8_8/vis_3999.png). | ![a](out_8_8/loss.png). | 
| 8            |   16   | ![a](out_8_16/out_8_16.gif). | ![a](out_8_16/vis_3999.png). | ![a](out_8_16/loss.png). | 
| 8            |   32   | ![a](out_8_32/out_8_32.gif). | ![a](out_8_32/vis_3999.png). | ![a](out_8_32/loss.png). | 
| 8            |   64   | ![a](out_8_64/out_8_64.gif). | ![a](out_8_64/vis_3999.png). | ![a](out_8_64/loss.png). | 
| 8            |   128   | ![a](out_8_128/out_8_128.gif). | ![a](out_8_128/vis_3999.png). | ![a](out_8_128/loss.png). | 
| 8            |   256   | ![a](out_8_256/out_8_256.gif). | ![a](out_8_256/vis_3999.png). | ![a](out_8_256/loss.png). | 
| 8            |   384   | ![a](out_8_384/out_8_384.gif). | ![a](out_8_384/vis_3999.png). | ![a](out_8_384/loss.png). | 
|--------------|-------|--------|--------|--------|
