import streamlit as st
import openai
import replicate
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np

# Set up API keys using Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]
replicate_api_key = st.secrets["replicate_api_key"]  # Set the Replicate API key here

# Initialize the OpenAI client
client = openai

# Initialize the Replicate client with the API key
replicate_client = replicate.Client(api_token=replicate_api_key)

# Load client secret from Streamlit secrets
client_secret = json.loads(st.secrets["google_drive_client_secret"])

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
                client_secret,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            flow.redirect_uri = client_secret["installed"]["redirect_uris"][0]

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

def list_images_in_folder(folder_id, service):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/'",
        pageSize=10, fields="files(id, name)").execute()
    items = results.get('files', [])
    return items

def load_image(file_id, service):
    request = service.files().get_media(fileId=file_id)
    img_data = request.execute()
    img = Image.open(BytesIO(img_data))
    return img

def analyze_images(images):
    # Dummy analysis - you can implement more complex analysis here
    colors = []
    for img in images:
        img = img.convert("RGB")
        img_resized = img.resize((50, 50))
        dominant_color = np.array(img_resized).mean(axis=(0, 1))
        colors.append(dominant_color)
    avg_color = np.mean(colors, axis=0)
    return f"Average color: {avg_color}"

def generate_image_with_replicate(prompt, analysis):
    # Use Replicate API to generate an image, incorporating analysis
    model_endpoint = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]  # Ensure this secret is correctly set
    output = replicate_client.run(
        model_endpoint,
        input={"prompt": f"{prompt} with {analysis}"}
    )
    return output[0]  # Assuming the first output is the image URL

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

if service:
    folder_id = st.text_input("Enter the Google Drive folder ID:")
    if folder_id:
        images_metadata = list_images_in_folder(folder_id, service)
        images = [load_image(img['id'], service) for img in images_metadata]
        for img in images:
            st.image(img)

        prompt = st.text_input("Enter your image creation prompt:")
        if prompt:
            # Analyze the images from Google Drive
            analysis = analyze_images(images)
            st.write("Image Analysis:", analysis)

            # Generate refined prompt using GPT-4o-mini
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Refine the following image creation prompt: {prompt} with analysis: {analysis}"}
                ],
                max_tokens=50
            )
            refined_prompt = completion.choices[0].message.content.strip()
            st.write("Refined Prompt:", refined_prompt)

            # Generate image using Replicate API
            generated_image_url = generate_image_with_replicate(refined_prompt, analysis)
            st.image(generated_image_url)
