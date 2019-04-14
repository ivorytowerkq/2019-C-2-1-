
# coding: utf-8

# In[118]:

import numpy as np


# In[119]:

from keras.models import Sequential,Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add


# In[120]:

from scipy.io import loadmat


# In[121]:

# Function to create to a series of CONV layers followed by Max pooling layer
def Convblock(channel_dimension, block_no, no_of_convs) :
    Layers = []
    for i in range(no_of_convs) :
        
        Conv_name = "conv"+str(block_no)+"_"+str(i+1)
        
        # A constant kernel size of 3*3 is used for all convolutions
        Layers.append(Convolution2D(channel_dimension,kernel_size = (3,3),padding = "same",activation = "relu",name = Conv_name))
    
    Max_pooling_name = "pool"+str(block_no)
    
    #Addding max pooling layer
    Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    
    return Layers


# In[122]:

def FCN_8_helper():
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = (512,512,3)))
    
    for l in Convblock(64,1,2) :
        model.add(l)
    
    for l in Convblock(128,2,2):
        model.add(l)
    
    for l in Convblock(256,3,3):
        model.add(l)
    
    for l in Convblock(512,4,3):
        model.add(l)
    
    for l in Convblock(512,5,3):
        model.add(l)
        
    model.add(Convolution2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6"))
      
    #Replacing fully connnected layers of VGG Net using convolutions
    model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))
    
    # Gives the classifications scores for each of the 21 classes including background
    model.add(Convolution2D(21,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))
    
    Conv_size = model.layers[-1].output_shape[2] #16 if image size if 512
    #print(Conv_size)
    
    model.add(Deconvolution2D(21,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score2"))
    
    # O = ((I-K+2*P)/Stride)+1 
    # O = Output dimesnion after convolution
    # I = Input dimnesion
    # K = kernel Size
    # P = Padding
    
    # I = (O-1)*Stride + K 
    Deconv_size = model.layers[-1].output_shape[2] #34 if image size is 512*512
    
    #print(Deconv_size)
    # 2 if image size is 512*512
    Extra = (Deconv_size - 2*Conv_size)
    
    #print(Extra)
    
    #Cropping to get correct size
    model.add(Cropping2D(cropping=((0,Extra),(0,Extra))))
    
    return model
    
    


# In[123]:

output = FCN_8_helper()
print(len(output.layers))


# In[124]:

output.summary()


# In[125]:

def FCN_8():
    fcn_8 = FCN_8_helper()
    #Calculating conv size after the sequential block
    #32 if image size is 512*512
    Conv_size = fcn_8.layers[-1].output_shape[2] 
    
    #Conv to be applied on Pool4
    skip_con1 = Convolution2D(21,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")
    
    #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
    Summed = add(inputs = [skip_con1(fcn_8.layers[14].output),fcn_8.layers[-1].output])
    
    #Upsampling output of first skip connection
    x = Deconvolution2D(21,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed)
    x = Cropping2D(cropping=((0,2),(0,2)))(x)
    
    
    #Conv to be applied to pool3
    skip_con2 = Convolution2D(21,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")
    
    #Adding skip connection which takes output og Max pooling layer 3 to current layer
    Summed = add(inputs = [skip_con2(fcn_8.layers[10].output),x])
    
    #Final Up convolution which restores the original image size
    Up = Deconvolution2D(21,kernel_size=(16,16),strides = (8,8),
                         padding = "valid",activation = None,name = "upsample")(Summed)
    
    #Cropping the extra part obtained due to transpose convolution
    final = Cropping2D(cropping = ((0,8),(0,8)))(Up)
    
    
    return Model(fcn_8.input, final)


# In[126]:

model = FCN_8()


# In[127]:

model.summary()


# In[128]:

from keras.utils import plot_model


# In[129]:

plot_model(model,"FCN-8with_shapes.png",show_shapes=True)


# In[130]:

#Loading weights from matlab file
data = loadmat('pascal-fcn8s-dag.mat', matlab_compatible=False, struct_as_record=False)
layers = data['layers']
params = data['params']
description = data['meta'][0,0].classes[0,0].description


# In[131]:

print(data.keys())


# In[132]:

print(layers.shape)


# In[133]:

#Inspecting layer names given in .mat file
for i in range(layers.shape[1]):
    print(i,
          str(layers[0,i].name[0]), str(layers[0,i].type[0]),
          [str(n[0]) for n in layers[0,i].inputs[0,:]],
          [str(n[0]) for n in layers[0,i].outputs[0,:]])


# In[134]:

#Inspecting filter and bias sizes 
for i in range(0, params.shape[1]-1, 2):
    print(i,
          str(params[0,i].name[0]), params[0,i].value.shape,
          str(params[0,i+1].name[0]), params[0,i+1].value.shape)


# In[135]:

params.shape


# In[136]:

for i in range(0, params.shape[1]):
    print(i,
          str(params[0,i].name[0]), params[0,i].value.shape)


# In[137]:

#Note : We are not transfering the weights of score4, score_pool3 and upsample because it doesn't follow the same 
# convention. The weights and biases for them are transferred below seperately

def copy_mat_of_keras(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (0, 1, 2, 3) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, 35, 2):
        matname = '_'.join(params[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            print(matname)
            kindex = kerasnames.index(matname)
            print('found : ', (str(matname), kindex))
            l_weights = params[0,i].value
            l_bias = params[0,i+1].value
            f_l_weights = l_weights.transpose(prmt)
            if False: # WARNING : this depends on "image_data_format":"channels_last" in keras.json file
                f_l_weights = np.flip(f_l_weights, 0)
                f_l_weights = np.flip(f_l_weights, 1)
            print(f_l_weights.shape, kmodel.layers[kindex].get_weights()[0].shape)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            print(f_l_weights.shape)
            print("layer")
            print(kmodel.layers[kindex].get_weights()[0].shape)
            print("layer")
            assert (l_bias.shape[1] == 1)
            print(l_bias[:,0].shape)
            print("bias")
            print(kmodel.layers[kindex].get_weights()[1].shape)
            print("bias")
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
        else:
            print('not found : ', str(matname))
    return kmodel


# In[138]:

model = copy_mat_of_keras(model)


# In[139]:

kerasnames = [lr.name for lr in model.layers]


# In[140]:

kerasnames


# In[141]:

#Getting the index of layer by name
kindex = kerasnames.index('score4')


# In[142]:

kindex


# In[143]:

l_weights = params[0,36].value


# In[144]:

l_weights.shape


# In[145]:

bias = np.zeros(21)


# In[146]:

#Giving weights and bias to corresponding layer
#Note : Bias is given as zero as it is absent in the .mat file
model.layers[27].set_weights([l_weights,bias])


# In[147]:

#Getting the index of layer by name
kindex = kerasnames.index('score_pool3')


# In[148]:

kindex


# In[149]:

l_weights = params[0,37].value


# In[150]:

l_weights.shape


# In[151]:

bias = params[0,38].value


# In[152]:

bias.shape


# In[153]:

bias = bias[:,0]


# In[154]:

bias.shape


# In[155]:

#Giving weights and bias to corresponding layer
model.layers[28].set_weights([l_weights,bias])


# In[156]:

#Getting the index of layer by name
kindex = kerasnames.index('upsample')


# In[157]:

kindex


# In[158]:

lweights = params[0,39].value


# In[159]:

lweights.shape


# In[160]:

bias = np.zeros(21)


# In[161]:

#Giving weights and bias to corresponding layer
#Note : Bias is given as zero as it is absent in the .mat file
model.layers[31].set_weights([lweights,bias])


# In[162]:

from skimage.io import imread, imsave


# In[163]:

import matplotlib.pyplot as plt
from PIL import Image
get_ipython().magic('matplotlib inline')


# In[164]:

import copy
import math
def prediction(kmodel, crpimg, transform=False):
	# INFO : crpimg should be a cropped image of the right dimension


	imarr = np.array(crpimg).astype(np.float32)

	if transform:
		imarr[:, :, 0] -= 129.1863
		imarr[:, :, 1] -= 104.7624
		imarr[:, :, 2] -= 93.5940
		
		aux = copy.copy(imarr)
		imarr[:, :, 0] = aux[:, :, 2]
		imarr[:, :, 2] = aux[:, :, 0]

	# imarr[:,:,0] -= 129.1863
	# imarr[:,:,1] -= 104.7624
	# imarr[:,:,2] -= 93.5940

	# imarr = imarr.transpose((2, 0, 1))
	imarr = np.expand_dims(imarr, axis=0)

	return kmodel.predict(imarr)


# In[165]:

import os
DATA_DIR=r"L:\附件1 图像序列"
for i, file in enumerate(os.listdir(DATA_DIR)):
    input_image_name = file
#     input_image = Image.open("images/" + input_image_name)
    im = Image.open(os.path.join(DATA_DIR, file))
# im = Image.open('TestImages/76.jpg') 
    im = im.crop((0,0,512,512)) # WARNING : manual square cropping
    im = im.resize((512,512))
    plt.imshow(np.asarray(im))
    crpim = im 
    preds = prediction(model, crpim, transform=False) 
    print(preds.shape)
    imclass = np.argmax(preds, axis=3)[0,:,:]
    imsave(file,imclass)
    print(type(imclass))
    print(len(imclass))
    print(imclass.shape)
    print(imclass.shape)

    plt.figure(figsize = (15, 7))
    plt.subplot(1,3,1)
    plt.imshow( np.asarray(crpim) )
    plt.subplot(1,3,2)
    plt.imshow( imclass )
    plt.subplot(1,3,3)
    plt.imshow( np.asarray(crpim) )
    masked_imclass = np.ma.masked_where(imclass == 0, imclass)
    #plt.imshow( imclass, alpha=0.5 )
    plt.imshow( masked_imclass, alpha=0.5 )
    plt.savefig(os.path.join('out', file))

