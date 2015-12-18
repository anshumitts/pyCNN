from functions import*
import variables as v
rng = np.random
rng.seed(0)
# contains components of each cel
class feature(object):
	"""docstring for feature"""
	def __init__(self,depth_of_input,kernal_size,Dimension_x):
		self.W 			=	2*rng.random((1,depth_of_input*kernal_size*kernal_size))-1
		self.B 			=	2*rng.random((1,1))-1
		self.gw 		=	np.zeros(self.W.shape)
		self.kernal_size=	kernal_size
		self.depth 		=	depth_of_input
		self.Dimension_x=	Dimension_x

	def output(self,Input):
		Output 		=	np.zeros(self.Dimension_x**2)
		row 	= 0
		col 	= 0
		for x in range(self.Dimension_x**2):
			input_val=Input[0][row:row+self.kernal_size,col:col+self.kernal_size].reshape(-1,1)
			for y in range(self.depth-1):
				img 		=	Input[y][row:row+self.kernal_size,col:col+self.kernal_size].reshape(-1,1)
				input_val	=	np.vstack((input_val,img))
			Output[x]		=	Net(self.W,input_val,self.B)[0][0]
			# if Output[x]<0:
			# 	Output[x]=0
			col+=1
			if col==self.Dimension_x:
				row+=1
				col=0
		Output 	=	norm(Output.reshape(self.Dimension_x,-1))
		return Output

	def weights(self):
		return self.W.reshape(-1,self.kernal_size)
	
	def update_weight(self,gw):
		self.gw 	=	v.alpha*gw + v.momt*self.gw
		self.W 		= 	(self.W - gw)

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
	def Out(self,Input):
		self.Input 	=	Input
		Outputs 	=	[]
		for neuron in self.features_map:
			Outputs.append(neuron.output(Input))
		return Outputs
	
	def error_map(self,Error):
		# shape 			=	error.shape[0]
		pre_size 			=	self.size+self.kernal_size-1
		depth				=	len(Error)
		Error_mat_gw 		=	[]
		error_features_mat	=	np.zeros((self.depth_of_input*self.kernal_size,self.kernal_size))
		Error_mat_pre_yr	=	[]
		error_pre_lyr		=	np.zeros((pre_size,pre_size))
		for t in range(depth):
			Error_mat_gw.append(error_features_mat)
		for q in range(self.depth_of_input):
			Error_mat_pre_yr.append(error_pre_lyr)
		
		i=0
		j=0
		for y in range(self.kernal_size**2):
			for x in range(depth):
				for w in range(self.depth_of_input):
					Error_mat_gw[x][w*self.kernal_size+i,j]+=(self.Input[w][i:i+self.size,j:j+self.size]*Error[x]).sum()	
			j+=1
			if j==self.kernal_size:
				j=0
				i+=1
		w_mat	=	[x.weights() for x in self.features_map]
		i=0
		j=0
		for f in range(self.size**2):
			for x in range(self.depth_of_input):
				for y in range(depth):
					Error_mat_pre_yr[x][i:i+self.kernal_size,j:j+self.kernal_size]+=w_mat[y][self.kernal_size*x:self.kernal_size*(x+1),:]*Error[y][i,j]		
			j+=1
			if j==self.size:
				j=0
				i+=1
		for x in range(depth):
			self.features_map[x].update_weight(Error_mat_gw[x].reshape(1,-1))
		return Error_mat_pre_yr
				
			

class max_pool:
	"""docstring for max_pool"""
	def __init__(self,features,kernal_size,slide,size):
		self.depth 		=	features
		self.kernal 	=	kernal_size
		self.slide		=	slide
		self.size 		=	int(size/slide)
		self.img_size	=	size
	def Out(self,Input):
		size_img=Input[0].shape[1]
		self.Input=Input
		Output=[]
		for x in range(self.depth):
			row=0
			col=0
			out= np.zeros(self.size**2)
			for y in range(self.size**2):
				out[y]=np.amax(Input[x][row:row+self.kernal,col:col+self.kernal])
				col+=self.slide
				if col+self.kernal>=size_img:
					row+=self.slide
					col=0
					if row+self.kernal>=size_img:
						break
			Output.append(out.reshape(self.size,-1))
		self.output=Output
		return Output
	def error_map(self,Error):
		Err 		=	self.Input[:]
		for x in range(self.depth):
			row=0
			col=0
			Err[x][:,:]	=	0
			for y in range(self.size**2):
				Err[x][row*self.slide:row*self.slide+self.kernal,col*self.slide:col*self.slide+self.kernal][self.Input[x][row*self.slide:row*self.slide+self.kernal,col*self.slide:col*self.slide+self.kernal]==self.output[x][row,col]]=Error[x][row,col]
				col+=1
				if col==self.size:
					row+=1
					col=0 
		return Err
		

# Final fully connected layer for Asigning classes 
class full_connected(object):
	"""docstring for full_connected"""
	def __init__(self, features,no_of_classes):
		self.no_of_classes	=	no_of_classes
		self.node_weight	=	2*rng.random((self.no_of_classes,features))-1
		self.gw 			=	np.zeros((self.no_of_classes,features))
		self.B 				=	2*rng.random((self.no_of_classes,1))-1
	def Out(self,Input):
		self.Input 		=	norm([[x[0,0]] for x in Input])
		# print self.Input
		self.Output 	= 	logistic(Net(self.node_weight,self.Input,self.B))
		return self.Output
	def error_map(self,error):
		output 				=	(self.node_weight*error).sum(axis=0)
		output 				=	[np.asarray([[x]]) for x in output]
		one 				=	np.ones(self.gw.shape)
		self.gw 			=	v.alpha*(one*(np.asarray(self.Input).reshape(1,-1)))*error + v.momt*self.gw
		# print self.gw
		self.node_weight-=self.gw
		return output