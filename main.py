from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

# --- 1. SETUP ---
# The full filename as it appears in the Replit file explorer

app = FastAPI(title="CowDex Breed Prediction API")

# Define the expected input format using Pydantic
class ImagePayload(BaseModel):
    # Base64 encoded image string
    image: str
MODEL_FILENAME = "cattle_breed_classifier_v1_37.h5"

# Load the trained TensorFlow model
# Make sure the filename matches the one you uploaded to Replit!
MODEL_FILENAME = "cattle_breed_classifier_v2_finetuned_96.h5" #<-- EDIT FILENAME
model = tf.keras.models.load_model(MODEL_FILENAME)
print(f"Model {MODEL_FILENAME} loaded successfully.")


# --- IMPORTANT: PASTE YOUR LABELS HERE ---
# Paste the output from train_generator.class_indices from your Colab notebook.
# It should look like: {'Angus': 0, 'Hereford': 1, 'Holstein': 2}
class_indices = {'Alambadi': 0, 'Amritmahal': 1, 'Ayrshire': 2, 'Banni': 3, 'Bargur': 4, 'Bhadawari': 5, 'Brown_Swiss': 6, 'Dangi': 7, 'Deoni': 8, 'Gir': 9, 'Guernsey': 10, 'Hallikar': 11, 'Hariana': 12, 'Holstein_Friesian': 13, 'Jaffrabadi': 14, 'Jersey': 15, 'Kangayam': 16, 'Kankrej': 17, 'Kasargod': 18, 'Kenkatha': 19, 'Kherigarh': 20, 'Khillari': 21, 'Krishna_Valley': 22, 'Malnad_gidda': 23, 'Mehsana': 24, 'Murrah': 25, 'Nagori': 26, 'Nili_Ravi': 27, 'Nimari': 28, 'Ongole': 29, 'Rathi': 30, 'Red_Sindhi': 31, 'Toda': 32, 'Umblachery': 33, 'Vechur': 34} #<-- EDIT LABELS
# We invert the dictionary to map from index to label name
labels = {v: k for k, v in class_indices.items()}
print(f"Labels loaded: {labels}")


# --- 2. API ENDPOINT ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the CowDex API! Go to /docs for the interactive API documentation."}


@app.post("/predict")
def predict_breed(payload: ImagePayload):
    """
    Takes a base64 encoded image string and returns the predicted cattle breed.
    """
    # 1. Decode the base64 image
    image_data = base64.b64decode(payload.image)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # 2. Preprocess the image for the model
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    # Normalize pixel values to be between -1 and 1 (as required by MobileNetV2)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Add a batch dimension
    data_for_prediction = np.expand_dims(normalized_image_array, axis=0)

    # 3. Make a prediction
    prediction = model.predict(data_for_prediction)

    # 4. Process the prediction
    predicted_index = np.argmax(prediction)
    predicted_breed = labels[predicted_index]
    confidence_score = float(prediction[0][predicted_index])

    print(f"Prediction successful: {predicted_breed} with confidence {confidence_score:.2f}")

    # 5. Return the result
    return {
        "breed": predicted_breed,
        "confidence": confidence_score
    }

# This part is not strictly needed for Replit as it has its own run command,
# but it's good practice for FastAPI apps.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)