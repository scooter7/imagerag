import streamlit as st
import openai
import replicate
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from PIL import Image
from io import BytesIO

# Set up API keys using Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]
replicate_api_key = st.secrets["replicate_api_key"]

# Initialize the OpenAI client
client = openai

# Initialize the Replicate client with the API key
replicate_client = replicate.Client(api_token=replicate_api_key)

# Initialize BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize emotion detection pipeline using the "j-hartmann/emotion-english-distilroberta-base" model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Load client secret from Streamlit secrets
client_secret = json.loads(st.secrets["google_drive_client_secret"])

@st.cache_data
def authenticate_google_drive():
    creds = None
    if "token" in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state["token"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
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

@st.cache_data
def list_images_in_folder(folder_id, service):
    st.write(f"DEBUG: Checking folder ID: {folder_id}")
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/'",
        pageSize=10, fields="files(id, name, mimeType, webViewLink)").execute()
    items = results.get('files', [])
    st.write(f"DEBUG: API response: {results}")
    return items

@st.cache_data
def load_image_cached(file_id, service):
    request = service.files().get_media(fileId=file_id)
    img_data = request.execute()
    img = Image.open(BytesIO(img_data))
    return img

def describe_image(image):
    try:
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        st.write(f"DEBUG: Generated description: {description}")
        return description
    except Exception as e:
        st.error(f"Error generating description: {e}")
        return None

def detect_emotions(description):
    try:
        predictions = emotion_model(description)
        if predictions:
            emotions = [f"{pred['label']} ({pred['score']:.2f})" for pred in predictions[0]]
            st.write(f"DEBUG: Detected emotions: {emotions}")
            return emotions
        else:
            st.write("DEBUG: No emotions detected.")
            return None
    except Exception as e:
        st.error(f"Error detecting emotions: {e}")
        return None

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

if service:
    folder_id = st.text_input("Enter the Google Drive folder ID:")
    if folder_id:
        images_metadata = list_images_in_folder(folder_id, service)
        
        if images_metadata:
            selected_image = st.selectbox(
                "Select an image to process", 
                [img['name'] for img in images_metadata]
            )

            if selected_image:
                img_metadata = next(img for img in images_metadata if img['name'] == selected_image)
                img = load_image_cached(img_metadata['id'], service)
                st.image(img)

                # Image description
                description = describe_image(img)
                if description:
                    # Emotion analysis based on the description
                    emotions = detect_emotions(description)
                    if emotions:
                        st.write("Detected Emotions:", ", ".join(emotions if emotions else []))
                    
                    # Store the image link
                    image_links = img_metadata['webViewLink']
                    
                    # Allow user to input a prompt for image creation
                    prompt = st.text_input("Enter your image creation prompt:")
                    if prompt:
                        combined_analysis = f"Image description: {description}. Detected emotions: {', '.join(emotions)}."
                        
                        proceed = st.button("Analyze and Generate Image")
                        if proceed:
                            # Generate refined prompt using GPT-4o-mini
                            completion = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": f"Refine the following image creation prompt: {prompt} with analysis: {combined_analysis}"}
                                ],
                                max_tokens=30  # Reduced token limit
                            )
                            refined_prompt = completion.choices[0].message.content.strip()
                            st.write("Refined Prompt:", refined_prompt)

                            # Generate image using Replicate API
                            generated_image_url = replicate_client.run(
                                st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"],
                                input={"prompt": refined_prompt}
                            )[0]
                            st.image(generated_image_url)

                            # Apply style transfer using one of the images from the folder
                            style_image = load_image_cached(img_metadata['id'], service)
                            style_transfer_result = replicate_client.run(
                                st.secrets["REPLICATE_STYLE_TRANSFER_MODEL_ENDPOINT"],
                                input={"content_image": generated_image_url, "style_image": style_image}
                            )[0]
                            st.image(style_transfer_result, caption="Image after style transfer")

                            # Explanation of how images informed the new generation
                            explanation = (
                                f"The new image was generated based on a refined prompt informed by the images in the selected folder. "
                                f"Textual descriptions of the images ({description}) and the detected emotions "
                                f"({', '.join(emotions)}) were used to refine the original prompt, shaping the content and mood of the new image. "
                                f"The style of the original image was then applied to the generated image using style transfer. "
                                f"You can view the original image that influenced this process here: [image]({image_links})."
                            )
                            st.write(explanation)
                        else:
                            st.write("DEBUG: Analysis and generation were not triggered.")
                    else:
                        st.write("DEBUG: No prompt provided.")
        else:
            st.write("DEBUG: No images found in the folder.")
    else:
        st.write("DEBUG: No folder ID provided.")
