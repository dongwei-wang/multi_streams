# brief introduction
This is the frame for my final project of CMSC691

I do images pixels conversion from RGB to gray scales
All images are stored at src/ directory which I specifiec at program
It can process any size of images only if you have enough big hard drives
Just store the images into src/, this frame work will process them

I implemented a frame with CUDA programming model with one GPU device, however we launch multistreams in one processing
I check the memory of CPU and GPU, and take the minimul one and reduce it a little bit as our memory upper boundary to collecting images
Just try to keep a safe line to avoid memory allocation break up

In each GPU implementation, I launch four streams to process the images data to overlap the memory copy latency.


# run the program
put the images into src/ directory

	make

	sh run

program will work


