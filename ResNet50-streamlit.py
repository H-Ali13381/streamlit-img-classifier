# 2220426 - Hasan Ali 

import streamlit as st
import requests

def main():
    st.title("ResNet50 Image Classifier", anchor=False)
    
    uploaded_file = st.file_uploader('Choose an image to classify:','jpg')
    
    run = st.button('Classify!')
    
    if run and uploaded_file is not None:
        
        with st.spinner('Working...'):

            #url = "http://127.0.0.1:8000/predict" --- for local deploy
            url = https://streamlit-img-classifier-backend.onrender.com/ # --- hosted location for api
            
            
            # This is needed to format the data files- alternatively, can use Request type in api for binary.
            files = {'image_file': ('filename.jpg', uploaded_file, 'image/jpeg')} 
            
            
            api = requests.post(url, files=files)
            result = api.json()
            
            st.write(result)
            
            #st.subheader(f"Classification: {result['Predicted class']}")
            

if __name__ == "__main__":
    main()
