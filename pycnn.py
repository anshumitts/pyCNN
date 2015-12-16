import cell_cnn as layer
import variables as v
import mnsit as imgs
import numpy as np
import cv2 as cv
import functions as F


# Input Layer
sizeOfImage		=	v.cellmatrix
connections		=	v.kernal
layer_0			=	layer.neuron_layer_cnn(1,5,8,sizeOfImage)
maxP_1			=	layer.max_pool(8,2,2,sizeOfImage-4)

#Hidden Layer 1

sizeOfImage		=	int((sizeOfImage-4)/2)
layer_0			=	layer.neuron_layer_cnn(8,5,16,sizeOfImage)
maxP_1			=	layer.max_pool(18,2,2,sizeOfImage-4)


#Hidden Layer 2

sizeOfImage		=	int((sizeOfImage-4)/2)
layer_0			=	layer.neuron_layer_cnn(8,5,16,sizeOfImage)
maxP_1			=	layer.max_pool(18,2,2,sizeOfImage-4)


#fully Connected Layer
sizeOfImage		=	int((sizeOfImage-4)/2)
Full_connected	=	layer.full_connected(sizeOfImage,v.classes)


#parsing data
Y_act 	=	np.zeros((v.classes),np.uint8) 
parsed	=	imgs.read(path=v.loc_databse)
length	=	len(parsed)	

for x in range(0):
	F.log((x+1)*100.0/length)
	Y_act[:]			=	0
	Y_act[parsed[x][1]]	=	1
	Y_cal				=	Full_connected.output(layer_2.neti(layer_1.neti(layer_0.neti(parsed[x][0]))))
	# imgs.show(parsed[x][0])
	print Y_act
	print Y_cal
	Y_cal				=	F.final_out([Y_cal])[0][0]
	print Y_cal
	cost_array 			= 	F.Calc_Error(Y_act,Y_cal)[0]
	layer_0.error_distibution(layer_1.error_distibution(layer_2.error_distibution(Full_connected.error_map(cost_array))))



	# print	Full_connected.output(layer_2.neti(layer_1.neti(layer_0.neti(parsed[0][0]))))
# for x in range(20):
# 	Y_act[:]			=	0
# 	Y_out[parsed[x][1]]	=	1
# 	Y_cal				=	Full_connected.output(layer_2.neti(layer_1.neti(layer_0.neti(parsed[x][0]))))
# 	print Y_out
# 	print (Y_cal/sum(Y_cal))