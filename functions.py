import theano.tensor as T
from theano import function
import cv2 as cv 
import numpy as np  
import math as mt
import logging

def log(percent,flag=1):
    logging.basicConfig(filename='percentage.log',level=logging.DEBUG)
    if flag==1:
        logging.info(percent)
    if flag==2:    
        logging.debug(percent)

e=mt.exp(1)

X 		= T.dmatrices('X')
B 		= T.dmatrices('B')
W 		= T.dmatrices('W')
Y 		= T.dmatrices('Y')
Y_act 	= T.dvector('Y_act')
Y_cal 	= T.dvector('Y_cal')
Error	= T.scalar(dtype=X.dtype)
P 		= T.scalar(dtype=X.dtype)

output 		= 	(1/(1+T.exp(-X)))
logistic 	= 	function([X], output)

# ouputs class for input sample
prediction 	=	X>0.5
final_out	=	function(inputs=[X], outputs=[prediction])

x 			= 	T.dot(W, Y)+B
Net 		= 	function([W,Y,B], x)

# Probability that target= 1
xent 		= 	(Y_act-Y_cal)*99999999999999#-Y_act*T.log(Y_cal) - (1-Y_act) * T.log(1-Y_cal) # Cross-entropy loss function
Calc_Error 	=	function(inputs=[Y_act,Y_cal], outputs=[xent]) 


# cost 		= 	Error + 0.01*(W**2).sum()# The cost to minimize
# gradient	= 	T.grad(cost, W)             # Compute the gradient of the cost
# grade 		= 	function(inputs=[Error,W], outputs=gradient)

# Raise each element of W to power P
raised		=	W**P
Power		=	function(inputs=[W,P], outputs=raised)



# Y_out1 		=	np.zeros((10,10),np.float64)
# Y_out2 		=	np.ones((10,10),np.float64)*0.5
# Y_out3		=	np.zeros((2,2))
# Y_out3[1,0]=1
# Y_out3[0,0]=1
# Y_out3[1,1]=1
# # # print Y_out1
# # # Y_cal 	=	Calc_Error(Y_out2,Y_out1)
# # # Y_cal[!]=0
# Y_out1[5:10,5:10][Y_out3==1]=0.5
# print Y_out1