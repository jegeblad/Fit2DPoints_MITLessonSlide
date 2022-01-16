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

For each combination of above parameters I train the model for 4000 epochs, and visualize every 100th epoch. The results looks like this:

| Inner layers | nodes | Animation     |  Final | Loss   |
|--------------|-------|--------|--------|--------|
| 2            |   2   | ![a](out_2_2/out_2_2.gif). | ![a](out_2_2/vis_3999.png). | ![a](out_2_2/loss.png). | 
|--------------|-------|--------|--------|--------|
