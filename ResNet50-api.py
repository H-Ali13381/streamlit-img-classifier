from fastapi import FastAPI, UploadFile, File
import uvicorn
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io


app = FastAPI(title="ResNet50 Classifier")

#Importing model
model = ResNet50(weights='imagenet')

@app.get("/")
def read_root():
    return {"Working": "true"}


    
@app.post("/predict")
async def classify(image_file: UploadFile = File(...)):
    """
    Classifies an image
    """

    image = await image_file.read()
    
    # Decode image and preprocess
    img = Image.open(io.BytesIO(image))#.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing image
    preprocessed_image = preprocess_input(img_array)
    
    # Making a prediction
    predictions = model.predict(preprocessed_image)
    
    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    
    predicted_class = str(decoded_predictions[1])
    certainty = f'{float(decoded_predictions[2])*100:.2f}%' 
    
    return {'Predicted class': predicted_class,
            'Certainty:': certainty}
    
    
if __name__ == "__main__":
    uvicorn.run("ResNet50-api:app",
                host='127.0.0.1',
                port=8000,
                reload=True)
