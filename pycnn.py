import cell_cnn as layer
import variables as v
import mnsit as imgs
import numpy as np
import functions as F


# Input Layer
sizeOfImage		=	v.size_of_img
connections		=	v.kernal
layer_0			=	layer.neuron_layer_cnn(1,5,8,sizeOfImage) #28
maxP1			=	layer.max_pool(8,2,2,sizeOfImage-4) #24

#Hidden Layer 1

sizeOfImage		=	int((sizeOfImage-4)/2) 
layer_1			=	layer.neuron_layer_cnn(8,5,16,sizeOfImage) #12
maxP2			=	layer.max_pool(16,2,2,sizeOfImage-4) #8


#Hidden Layer 2

sizeOfImage		=	int((sizeOfImage-4)/2)
layer_2			=	layer.neuron_layer_cnn(16,4,32,sizeOfImage) #4


#fully Connected Layer
sizeOfImage		=	int((sizeOfImage-3))
Full_connected	=	layer.full_connected(32,v.classes)


#parsing data
Y_act 	=	np.zeros((v.classes,1),np.uint8) 
parsed	=	imgs.read(path=v.loc_databse)
length	=	len(parsed)	
x=0
for x in range(length):
	F.log((x+1)*100.0/length)
	Y_act[:,0]				=	0
	Y_act[parsed[x][1],0]	=	1
	print x
	img 	=	[parsed[x][0].astype(float)/2550]
	img 	=	layer_0.Out(img)
	img 	=	maxP1.Out(img)
	img 	=	layer_1.Out(img)
	img 	=	maxP2.Out(img)
	img 	=	layer_2.Out(img)
	Y_cal 	=	Full_connected.Out(img)
	error 	= 	Full_connected.error_map(Y_act - Y_cal)
	error 	= 	layer_2.error_map(error)
	error 	= 	maxP2.error_map(error)
	error 	= 	layer_1.error_map(error)
	error 	= 	maxP1.error_map(error)
	error 	= 	layer_0.error_map(error)
	# imgs.show(parsed[x][0])
	print Y_act
	print Y_cal