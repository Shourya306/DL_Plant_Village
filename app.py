import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


nav = st.sidebar.radio('Models',['Potato_Classifer','Pepper_Classifier'])
if nav == 'Potato_Classifer':
    st.title('Model_info')
    st.image('https://kj1bcdn.b-cdn.net/media/60759/potato25.jpg?width=1200')
    st.markdown('''The model takes in potato plants leaf as input and predicts one of the following:
                <br>
                1. **Late Blight(Unhealthy)**
                <br>
                2. **Early Blight(Unhealthy)**
                <br>
                3. **Healthy**''',True)
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = keras.models.load_model('/Users/shouryachalla/Desktop/potato_classifier/best_model')
        return model
    with st.spinner('Model is being loaded....'):
        model = load_model()
    file = st.file_uploader('Image',type=['jpg','png'],accept_multiple_files = False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data,model):
        size = (256,256)
        image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
    
    if file is None:
        st.text('Please upload the image')
    else:
        image = Image.open(file)
        st.image(image)
        predictions = import_and_predict(image,model)
        class_names = ['Early_blight','Late_blight','healthy']
        result = class_names[np.argmax(predictions)]
        st.success('The prediction is ' + result)

else:
    st.title('Model_info')
    st.image('https://c0.wallpaperflare.com/preview/234/402/419/plant-ghost-pepper-heat-garden.jpg')
    st.markdown('''The model takes in Pepper leaf as input and predicts one of the following:
                <br>
                1. **Bacterial Leaf**
                <br>
                2.**Healthy leaf**
                <br>
    ''',True)
    @st.cache(allow_output_mutation=True)
    def load_model():
        pep_model = keras.models.load_model('/Users/shouryachalla/Desktop/potato_classifier/best_pepper_model')
        return pep_model
    with st.spinner('The model is being loaded.....'):
        model = load_model()
    file = st.file_uploader('Image',accept_multiple_files = False,type = ['jpg','png'])


    def import_and_predict(image,model):
        size = (256,256)
        image = ImageOps.fit(image,size,Image.ANTIALIAS)
        image = np.asarray(image)
        image = image[np.newaxis,...]
        predictions = model.predict(image)
        return predictions
    if file is None:
        st.text('Please Upload the Image')
    else:
        image = Image.open(file)
        st.image(image)
        predictions = import_and_predict(image,model)
        if predictions > 0.5:
            st.text('The prediction is Healthy')
        else:
            st.text('The prediction is Bacterial')

