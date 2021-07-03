from glob import glob
import os
import shutil
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

def folder_structure_creator(stringpathlist):
    for stringpath in stringpathlist:
        if os.path.exists(stringpath):
            print('Parent folder already created')
        else:
            os.makedirs(stringpath)
            print('Parent folder created')

    return None

def input_file_copy_to_targets(source,target,typ,traincount,valcount,testcount,gen_image_per_original):
    files=glob(source+'/*.png')
    random.shuffle(files)
    i=1
    for elem in files:
        file=elem.split('/')[-1]
        
        if i<=traincount:
            newtarget=target+'/train/'+typ+'/'            
            shutil.copy(source+'/'+file,newtarget)
            #print('Original file',file)
            if gen_image_per_original > 1:
                new_image_create(newtarget+file,newtarget,gen_image_per_original)
            i=i+1
            #break
        elif i>traincount and i<=(traincount+valcount):
            newtarget=target+'/val/'+typ+'/'
            shutil.copy(source+'/'+file,newtarget)
            i=i+1
        else:
            newtarget=target+'/test/'+typ+'/'
            shutil.copy(source+'/'+file,newtarget)
            i=i+1
    return None
            

def new_image_create(imagesource,target,numofimg):
    img=load_img(imagesource)
    img=img_to_array(img)
    img=img.reshape((1,)+img.shape)
    #print(img.shape)
    datagen=ImageDataGenerator(brightness_range=[0.5,1.0],
                               zoom_range=0.2,
                               horizontal_flip=True)
    i=0
    for batch in datagen.flow(img,save_to_dir=target,batch_size=1,save_format='png'):
        if i==numofimg-1:
            #print('Total generated',i)
            break
        i=i+1
    return None
        


    
    

paths=[
'Data/train/COVID',
'Data/train/Normal',
'Data/train/lungopt',
'Data/train/pneumonia',
'Data/val/COVID',
'Data/val/Normal',
'Data/val/lungopt',
'Data/val/pneumonia',
'Data/test/COVID',
'Data/test/Normal',
'Data/test/lungopt',
'Data/test/pneumonia']

folder_structure_creator(paths)

source='../COVID-19_Radiography_Dataset/COVID'
input_file_copy_to_targets(source,'Data','COVID',3300,200,116,2)

print('COVID data copied')

source='../COVID-19_Radiography_Dataset/Lung_Opacity'
input_file_copy_to_targets(source,'Data','lungopt',5800,200,12,1)

print('Lung_Opacity data copied')

source='../COVID-19_Radiography_Dataset/Normal'
input_file_copy_to_targets(source,'Data','Normal',6800,200,3192,1)

print('Normal data copied')

source='../COVID-19_Radiography_Dataset/Viral Pneumonia'
input_file_copy_to_targets(source,'Data','pneumonia',1100,200,45,3)

print('pneumonia data copied')



    








