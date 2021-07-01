from keras.preprocessing.image import load_img,img_to_array,array_to_img
from keras.models import load_model
from glob import glob
import numpy as np
from tensorflow.keras.models import Model
import tensorflow
from matplotlib import pyplot as plt
from matplotlib import cm




def image_prediction(model,image):
    classes=['COVID','Normal','Lung Opacity','Viral Pneumonia']
    
    image=load_img(image)
    image=img_to_array(image)
    image=image/255
    #print(image.shape)
    #image=image.reshape((1,)+image.shape())
    
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    #print(image.shape)
    image=tensorflow.image.resize_with_pad(image,299,299)
    #print(image.shape)
    result=model.predict(image)
    print(classes[np.argmax(result)])
    return classes[np.argmax(result)]

def grad_cam_heatmap(model,last_conv_layer_name,image):
    grad_model=Model([model.inputs],[model.get_layer(last_conv_layer_name).output,model.output])
    image=load_img(image)
    image=img_to_array(image)
    image=image/255
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image=tensorflow.image.resize_with_pad(image,299,299)
    with tensorflow.GradientTape() as tape:
        last_conv_output,preds=grad_model(image)
        pred_index=tensorflow.argmax(preds[0])
        class_channel=preds[:,pred_index]
        #print('last_conv_output',last_conv_output.shape)
        #print('preds',preds)
        #print('pred_index',pred_index)
        #print('class_channel',class_channel)
        grad=tape.gradient(class_channel,last_conv_output)
        pooled_grads=tensorflow.reduce_mean(grad,axis=(0,1,2))
        last_conv_output=last_conv_output[0]
        heatmap=last_conv_output @ pooled_grads[...,tensorflow.newaxis]
        heatmap=tensorflow.squeeze(heatmap)
        heatmap=tensorflow.maximum(heatmap,0)/tensorflow.math.reduce_max(heatmap)

    return heatmap.numpy()



def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))
    return None




    
    
model=load_model('Best_model.h5')
model1=load_model('../web/Best_model.h5')
model.summary()
imagepath='./Data/test/lungopt/Lung_Opacity-3940.png'
#imagepath='Sample.jpeg'
result=image_prediction(model,imagepath)
result=image_prediction(model1,imagepath)
#heatmap=grad_cam_heatmap(model,'conv2d_14',imagepath)
#plt.matshow(heatmap)
#plt.show()
#save_and_display_gradcam(imagepath, heatmap,cam_path=result+'.jpg')

##imagepath='./Data/test/COVID/'

##images=glob(imagepath+'*.png')
##i=0
##for elem in images:
##    print(elem)
##    image_prediction(model,elem)
##    if i==10:
##        break
##    i=i+1
##    
##print('================Lungopt====================')
##imagepath='./Data/test/lungopt/'
##images=glob(imagepath+'*.png')
##i=0
##for elem in images:
##    print(elem)
##    image_prediction(model,elem)
##    if i==10:
##        break
##    i=i+1
##
##print('=================Normal===================')
##imagepath='./Data/test/Normal/'
##images=glob(imagepath+'*.png')
##i=0
##for elem in images:
##    print(elem)
##    image_prediction(model,elem)
##    if i==10:
##        break
##    i=i+1
##
##print('==================Pneumonia==================')
##imagepath='./Data/test/pneumonia/'
##images=glob(imagepath+'*.png')
##i=0
##for elem in images:
##    print(elem)
##    image_prediction(model,elem)
##    if i==10:
##        break
##    i=i+1
