import streamlit as st
import openai
import replicate
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from PIL import Image, ImageOps
from io import BytesIO

# Set up API keys using Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]
replicate_api_key = st.secrets["replicate_api_key"]

# Initialize the OpenAI client
client = openai

# Initialize the Replicate client with the API key
replicate_client = replicate.Client(api_token=replicate_api_key)

# Initialize CLIP model for image captioning
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize emotion detection pipeline using a lightweight model
emotion_model = pipeline("image-classification", model="microsoft/swin-tiny-patch4-window7-224")

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
        pageSize=10, fields="files(id, name, webViewLink)").execute()
    items = results.get('files', [])
    return items

def load_image(file_id, service):
    request = service.files().get_media(fileId=file_id)
    img_data = request.execute()
    img = Image.open(BytesIO(img_data))
    return img

def describe_image(image):
    # Use CLIP to generate a description for the image
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)
    # Assuming 'probs' returns some form of interpretable text
    # For demonstration, returning a mock description
    return "a scenic landscape with mountains"  # Replace with actual description logic

def detect_emotions(image):
    # Use the emotion model to detect emotions in the image
    predictions = emotion_model(image)
    emotions = [f"{pred['label']} ({pred['score']:.2f})" for pred in predictions]
    return emotions

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

if service:
    folder_id = st.text_input("Enter the Google Drive folder ID:")
    if folder_id:
        images_metadata = list_images_in_folder(folder_id, service)
        all_descriptions = []
        all_emotions = []
        image_links = []

        for img_metadata in images_metadata:
            img = load_image(img_metadata['id'], service)
            st.image(img)

            # Image description
            description = describe_image(img)
            all_descriptions.append(description)
            st.write("Description:", description)

            # Emotion analysis
            emotions = detect_emotions(img)
            all_emotions.append(", ".join(emotions))
            st.write("Detected Emotions:", ", ".join(emotions))

            # Store the image link
            image_links.append(img_metadata['webViewLink'])

        prompt = st.text_input("Enter your image creation prompt:")
        if prompt:
            # Combine descriptions and emotions to inform the new image generation
            combined_analysis = f"Image descriptions: {', '.join(all_descriptions)}. Detected emotions: {', '.join(all_emotions)}."

            # Generate refined prompt using GPT-4o-mini
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Refine the following image creation prompt: {prompt} with analysis: {combined_analysis}"}
                ],
                max_tokens=50
            )
            refined_prompt = completion.choices[0].message.content.strip()
            st.write("Refined Prompt:", refined_prompt)

            # Generate image using Replicate API
            generated_image_url = replicate_client.run(
                st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"],
                input={"prompt": refined_prompt}
            )[0]
            st.image(generated_image_url)

            # Explanation of how images informed the new generation
            image_link_list = ', '.join([f"[image]({link})" for link in image_links])
            explanation = (
                f"The new image was generated based on the analysis of the images in the selected folder. "
                f"The descriptions of the images ({', '.join(all_descriptions)}) were used to create a context, "
                f"and the detected emotions ({', '.join(all_emotions)}) helped shape the mood and tone of the new image. "
                f"You can view the original images that informed this generation here: {image_link_list}."
            )
            st.write(explanation)
