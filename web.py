from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow
from keras.models import load_model
from tempfile import NamedTemporaryFile
import tensorflow.compat.v1.keras.backend as K


@st.cache(allow_output_mutation=True)
def loadmodel():
    model=load_model('Best_model.h5')
    print('loadmodel function invoked and Model loaded')
    session=K.get_session()
    return model,session

st.write('My First App')
upload_file=st.file_uploader("Upload your X-RAY image")
if upload_file is not None:
    print('====================================')
    pngfile=Image.open(upload_file).convert('RGB')
    image=img_to_array(pngfile)
    image=image/255
    print(image.shape)
    image=image.reshape(1,image.shape[0],image.shape[1],3)
    print(image.shape)
    image=tensorflow.image.resize_with_pad(image,299,299)
    model,session=loadmodel()
    K.set_session(session)
    result=model.predict(image)
    
    print(result)
    classes=['COVID','Normal','Lung Opacity','Viral Pneumonia']
    print(classes[np.argmax(result)])
    st.write('Findings:'+str(classes[np.argmax(result)]))

