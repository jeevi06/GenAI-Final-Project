import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import google.generativeai as genai
from langchain_google_genai.llms import GoogleGenerativeAI


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Read the API key from the text file
def get_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Get the API key from the txt file
GEMINI_API_KEY = "AIzaSyAb5YQAtOOqy7fgM4jRXGd-Ghs6VKlN-ZA"
#get_api_key_from_file("C:/Users/aarsh/OneDrive/Desktop/AI Assistant/gemini_api_key.txt")  
#os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


print("Google API Key initialized successfully")

# Initialize Google Generative AI model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)


# Set up Streamlit page
st.set_page_config(page_title="VisionMate", layout="wide")
st.title("VisionMate - AI Powered Solution for Assisting Visually Impaired Individuals")
st.sidebar.title("VisionMate Features")
st.sidebar.markdown("""
Welcome to **VisionMate**! Select a service below to assist you with various tasks:

- **Scene Understanding**: Understand the content of images.
- **Text Extraction**: Extract text from images using OCR.
- **Text-to-Speech**: Convert extracted text into speech.
- **Object Detection**: Identify objects in images using AI.


""")


def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])  # Use image data here
    return response.text


def extract_text_from_image(image):
    """Extracts text from an image"""
    text = pytesseract.image_to_string(image)
    return text


# Initialize Text-to-Speech engine
speech = pyttsx3.init()

def text_to_speech(text):
    """Converts text to speech."""
    speech.say(text)
    speech.runAndWait()
    
    
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

def object_detection(image):
    """Performs object detection on the uploaded image using YOLO."""
    # Load pre-trained YOLOv5 model
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
    model = YOLO('yolov5s.pt')
    
    img = np.array(image)
    results = model(img)  # Inference

    # Convert results to an image with bounding boxes
    annotated_image = results[0].plot()  # Annotate the image

    # Convert back to PIL format for display
    img_with_boxes = Image.fromarray(annotated_image)

    return img_with_boxes, results[0].boxes.xywh.numpy()
    #print(model)
    
    
    #img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    
    #results = model(img)
    
   
    #results.render()  
    
    # Convert back to PIL format for display
    #img_with_boxes = Image.fromarray(results.ims[0])
    
   # return img_with_boxes, results.pandas().xywh



# Main app functionality
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Buttons for functionalities
col1, col2, col3,col4 = st.columns(4)
scene_button = col1.button("Scene Description")
ocr_button = col2.button("Extract Text")
tts_button = col3.button("Text-to-Speech")
od_button = col4.button("Object Detection")




input_prompt = """
"As an AI assistant, your role is to assist visually impaired individuals by describing the content of the uploaded image. Please provide the following:

A list of objects identified in the image along with their functions.
A comprehensive description of the scene.
Recommendations for actions or safety precautions that could be helpful for the visually impaired."
"""


if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Creating a detailed description of the scene"):
            response = generate_scene_description(input_prompt, image_data)
            st.subheader("Scene Description")
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from image."):
            text = extract_text_from_image(image)
            if text.strip():
                st.subheader("Extracted Text")
                st.write(text)
            else:
                st.warning("No text found in the image.")

    if tts_button:
        with st.spinner("Converting text to speech."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("Text-to-Speech Conversion Finished!")
            else:
                st.warning("No text found in the image for conversion.")

    if od_button:
        with st.spinner("Performing object detection..."):
            img_with_boxes, results = object_detection(image)
            st.subheader("Detected Objects")
            st.image(img_with_boxes, caption="Image with Detected Objects", use_column_width=True)
            st.write(results)