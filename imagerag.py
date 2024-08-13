import streamlit as st
import openai
import replicate
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from PIL import Image
from io import BytesIO

# Set up API keys using Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]
replicate.api_key = st.secrets["replicate_api_key"]

# Google Drive Authentication
client_secret = json.loads(st.secrets["google_drive_client_secret"])

def authenticate_google_drive():
    creds = None
    if "token" in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state["token"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(client_secret, scopes=['https://www.googleapis.com/auth/drive'])
            flow.redirect_uri = "https://imageragpy-ddyawcsqcvxq3tzjkjfub2.streamlit.app/"
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.write("Please go to the following URL and authorize access:")
            st.write(auth_url)
            auth_code = st.text_input("Enter the authorization code here:")
            if auth_code:
                creds = flow.fetch_token(code=auth_code)
                st.session_state["token"] = json.loads(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service

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

def generate_image_with_replicate(prompt):
    # Use Replicate API to generate an image
    model = replicate.models.get("your-model-id")
    output = model.predict(prompt=prompt)
    return output[0]  # Assuming the first output is the image URL

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

folder_id = st.text_input("Enter the Google Drive folder ID:")
if folder_id:
    images = list_images_in_folder(folder_id, service)
    selected_image = st.selectbox("Select an image", [img['name'] for img in images])
    
    if selected_image:
        img_id = next(img['id'] for img in images if img['name'] == selected_image)
        img = load_image(img_id, service)
        st.image(img)

    prompt = st.text_input("Enter your image creation prompt:")
    if prompt:
        # Generate refined prompt using GPT-4o-mini
        response = openai.Completion.create(
            engine="gpt-4o-mini",
            prompt=f"Refine the following image creation prompt: {prompt}",
            max_tokens=50
        )
        refined_prompt = response.choices[0].text.strip()
        st.write("Refined Prompt:", refined_prompt)

        # Generate image using Replicate API
        generated_image_url = generate_image_with_replicate(refined_prompt)
        st.image(generated_image_url)
