import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from PIL import Image, ImageOps
from io import BytesIO

# Initialize CLIP model for image captioning
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize emotion detection pipeline using a lightweight model
emotion_model = pipeline("image-classification", model="microsoft/swin-tiny-patch4-window7-224")

def authenticate_google_drive():
    creds = None
    if "token" in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state["token"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use InstalledAppFlow for the "installed" configuration
            flow = InstalledAppFlow.from_client_config(
                json.loads(st.secrets["google_drive_client_secret"]),
                scopes=['https://www.googleapis.com/auth/drive']
            )
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            auth_url, _ = flow.authorization_url(prompt='consent')
            st.write("Please go to the following URL and authorize access:")
            st.write(auth_url)

            auth_code = st.text_input("Enter the authorization code here:")
            if auth_code:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                st.session_state["token"] = json.loads(creds.to_json())

    if creds:
        service = build('drive', 'v3', credentials=creds)
        return service
    else:
        st.error("Failed to authenticate with Google Drive.")
        return None

def load_image(file_id, service):
    request = service.files().get_media(fileId=file_id)
    img_data = request.execute()
    img = Image.open(BytesIO(img_data))
    return img

def describe_image(image):
    try:
        # Use CLIP to generate a description for the image
        inputs = clip_processor(images=image, return_tensors="pt")
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)
        # For debugging, output the probabilities to see if it’s working
        st.write(f"DEBUG: Probabilities from CLIP: {probs}")
        description = "a scenic landscape with mountains"  # Replace with actual logic
        st.write(f"DEBUG: Generated description: {description}")
        return description
    except Exception as e:
        st.error(f"Error generating description: {e}")
        return None

def detect_emotions(image):
    try:
        # Use the emotion model to detect emotions in the image
        predictions = emotion_model(image)
        emotions = [f"{pred['label']} ({pred['score']:.2f})" for pred in predictions]
        st.write(f"DEBUG: Detected emotions: {emotions}")
        return emotions
    except Exception as e:
        st.error(f"Error detecting emotions: {e}")
        return None

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

if service:
    folder_id = st.text_input("Enter the Google Drive folder ID:")
    if folder_id:
        images_metadata = service.files().list(
            q=f"'{folder_id}' in parents and mimeType contains 'image/'",
            pageSize=1,  # Limit to 1 image for debugging
            fields="files(id, name, webViewLink)"
        ).execute().get('files', [])

        if images_metadata:
            img_metadata = images_metadata[0]  # Get the first image
            img = load_image(img_metadata['id'], service)
            st.image(img)

            # Image description
            description = describe_image(img)
            st.write("Description:", description)

            # Emotion analysis
            emotions = detect_emotions(img)
            st.write("Detected Emotions:", ", ".join(emotions))
