import theano.tensor as T
from theano import function
import cv2 as cv 
import copy, numpy as np
import math as mt
import logging

def log(percent,flag=1):
    logging.basicConfig(filename='percentage.log',level=logging.DEBUG)
    if flag==1:
        logging.info(percent)
    if flag==2:    
        logging.debug(percent)

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

output1 	= 	T.tanh(-X)
norm	 	= 	function([X], output1)

# ouputs class for input sample
prediction 	=	X>0.5
final_out	=	function(inputs=[X], outputs=[prediction])

x 			= 	T.dot(W, Y)+B
Net 		= 	function([W,Y,B], x)

# Probability that target= 1
xent 		= 	(Y_act-Y_cal)
Calc_Error 	=	function(inputs=[Y_act,Y_cal], outputs=[xent]) 

# Raise each element of W to power P
raised		=	W**P
Power		=	function(inputs=[W,P], outputs=raised)
