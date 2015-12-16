from functions import*
import variables as v
rng = np.random

# contains components of each cell
class feature(object):
	"""docstring for feature"""
	def __init__(self,depth_of_input,kernal_size,Dimension_x):
		self.W 			=	rng.randn(depth_of_input*kernal_size*kernal_size)
		self.B 			=	rng.randn(1,1)
		self.kernal_size=	kernal_size
		self.depth 		=	depth_of_input
		self.Dimension_x=	Dimension_x

	def output(self,Input):
		Output 		=	np.zeros(self.Dimension_x**2,np.float64)
		row 	= 0
		col 	= 0
		for x in range(self.Dimension_x**2):
			img 		=	(Input[:][row:row+self.kernal_size,col:col+self.kernal_size].reshape(1,-1))
			input_val	=	np.vstack(img)
			Output[x]	=	Net(self.W,input_val,self.B)[0][0]
			col+=1
			if col==self.Dimension_x:
				row+=1
				col=0
		Output 	=	Output.reshape(self.Dimension_x,-1)
		return Output

	def weights():
		return self.W.reshape(self.kernal_size,-1)
	def update_weight(self,gw):
		W_old	=	self.W.reshape(self.kernal_size,-1)
		self.W 	= 	(self.W - v.alpha*gw)
		return	W_old*error

# contains features of each layer
class neuron_layer_cnn(object):
	"""docstring for feature_layer"""
	def __init__(self,depth_of_input,kernal_size,features,Input_shape=0):
		self.size = Input_shape - kernal_size + 1
		def store_features(features,kernal_size,size):
			output=[]
			for x in range(features): 
				output.append (feature(depth_of_input,kernal_size,size))
			return output
		self.features_map	=	store_features(features,kernal_size,self.size)
		self.features 		=	features
		self.kernal_size 	=	kernal_size
		self.depth_of_input =	depth_of_input
	#Input =[[depth0],[depth1],[depth2]]
	def output(self,Input):
		self.Input 	=	Input
		Outputs 	=	[]
		Outputs[:]	=	self.features_map[:].output(Input)
		return Outputs
	
	def error_distibution(self,Error):
		# shape 			=	error.shape[0]
		depth				=	len(Error)
		Error_mat_gw 		=	[]
		error_features_mat	=	np.zeros((self.kernal_size,self.kernal_size),np.float64)
		Error_mat_pre_yr	=	[]
		error_pre_lyr		=	np.zeros((pre_size,pre_size),np.float64)
		for t in range(depth):
			Error_mat_gw.append(error_features_mat)
		for q in range(self.depth_of_input):
			Error_mat_pre_yr.append(error_pre_lyr)
		
		i=0
		j=0
		for y in range(self.depth_of_input*(self.kernal_size**2)):
			for x in range(depth):
				for w in range(self.depth_of_input):
					Error_mat_gw[x][w*self.kernal_size+i,j]+=(self.Input[w][i:i+self.size,j:j+self.size]*Error[x][i:i+self.size,j:j+self.size]).sum()	
			j+=1
			if j==self.kernal_size:
				j=0
				i+=1
		w_mat	=	self.features_map[:].weights()
		i=0
		j=0
		for f in range(size**2):
			for x in range(self.depth_of_input):
				for y in range(depth):
					Error_mat_pre_yr[x][i:i+self.kernal_size,j:j+self.kernal_size]+=w_mat[y][self.depth_of_input*x:self.depth_of_input*x+self.kernal_size,:]*Error[y][i,j]		
			j+=1
			if j==size:
				j=0
				i+=1
		for x in range(depth):
			self.features_map[x].update_weight(Error_mat_gw[x])
		return Error_mat_pre_yr
				
			

class max_pool(object):
	"""docstring for max_pool"""
	def __init__(self,features,kernal_size,slide,size):
		self.depth 		=	features
		self.kernal 	=	kernal_size
		self.slide		=	slide
		self.size 		=	int(size/slide)
		self.img_size	=	size
	def output(self,Input):
		size_img=Input[0].shape[1]
		self.Input=Input
		Ouput=[]
		for x in range(self.depth):
			row=0
			col=0
			out= np.zeros(size**2,np.float64)
			for y in range(size**2):
				out[y]=max(Input[x][row:row+self.kernal_size,col:col+self.kernal_size].reshape[1,-1][0])
				col+=self.slide
				if col>=size_img:
					row+=self.slide
					col=0
			Ouput.append(out.reshape(self.size,-1))
		self.output=Output
		return Output
	def error_map(self,Error):
		Err 		=	self.Input[:].copy()
		Err[:][:,:]	=	0
		for x in range(self.depth):
			row=0
			col=0
			for y in range(self.img_size**2):
				Err[x][row:row+self.kernal_size,col:col+self.kernal_size][self.Input[x][row:row+self.kernal_size,col:col+self.kernal_size]==self.output[x][row,col]]=Error[x][row,col]
				col+=self.slide
				if col>=size_img:
					row+=self.slide
					col=0
		return Err
		
		

# Final fully connected layer for Asigning classes 
class full_connected(object):
	"""docstring for full_connected"""
	def __init__(self, features,no_of_classes):
		self.no_of_classes	=	no_of_classes
		self.node_weight	=	rng.randn(self.no_of_classes,features)
		self.B 				=	rng.randn(self.no_of_classes,1)
		self.cell_mat		=	mt.sqrt(features)
	def output(self,Input):
		self.Input 			=	Input.reshape(-1,1)
		self.Ouput 			= 	np.asarray(logistic(Net(self.node_weight,self.Input,self.B))).reshape(1,-1)[0]
		return self.Ouput
	def error_map(self,error):
		Dirivative_arr	=	self.Ouput*(1-self.Ouput)
		Errot_mat		=	np.zeros(self.node_weight.shape)
		for x in range(self.no_of_classes):
			W 						=	self.node_weight[x,:]
			error[x]				=	Dirivative_arr[x]*error[x]
			gw 						=	grade(error[x],[W])[0]
			gb 						=	grade(error[x],[self.B[x,:]])[0]
			Errot_mat[x,:]			=	W*error[x] 
			self.node_weight[x,:] 	=	W - v.alpha*gw
			self.B[x,:] 			=	self.B[x,:] - v.alpha*gb
<<<<<<< HEAD
		return	Errot_mat.sum(axis=0)
=======
		return	Errot_mat.sum(axis=0)
>>>>>>> 84711d03199036f305ba4c2427f3f51cf7f9b927
