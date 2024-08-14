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
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import requests  # Added import for requests

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
def authenticate_google_drive(auth_code=None):
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

            if not auth_code:
                auth_url, _ = flow.authorization_url(prompt='consent')
                st.write("Please go to the following URL and authorize access:")
                st.write(auth_url)
            else:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                st.session_state["token"] = json.loads(creds.to_json())

    if creds:
        service = build('drive', 'v3', credentials=creds)
        return service
    else:
        return None

# Check if the user is already authenticated
if "token" not in st.session_state:
    auth_code = st.text_input("Enter the authorization code here:")
    service = authenticate_google_drive(auth_code)
    if service:
        st.success("Successfully authenticated with Google Drive.")
else:
    service = authenticate_google_drive()
    st.success("You are already authenticated with Google Drive.")

@st.cache_data
def list_images_in_folder(_service, folder_id):
    st.write(f"DEBUG: Checking folder ID: {folder_id}")
    results = _service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/'",
        pageSize=10, fields="files(id, name, mimeType, webViewLink)").execute()
    items = results.get('files', [])
    st.write(f"DEBUG: API response: {results}")
    return items

@st.cache_data
def load_image_cached(_service, file_id):
    request = _service.files().get_media(fileId=file_id)
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

class StyleTransferModel:
    def __init__(self, content_img, style_img, device):
        self.device = device
        self.content_img = self.image_loader(content_img).to(self.device, torch.float)
        self.style_img = self.image_loader(style_img).to(self.device, torch.float)
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000000

        self.model, self.style_losses, self.content_losses = self.get_style_model_and_losses()

    def image_loader(self, image):
        loader = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        image = loader(image).unsqueeze(0)
        return image

    def get_style_model_and_losses(self):
        cnn = self.cnn
        content_losses = []
        style_losses = []

        model = nn.Sequential()
        i = 0  
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = nn.MSELoss()(model, target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target = model(self.style_img).detach()
                style_loss = nn.MSELoss()(model, target)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], nn.MSELoss):
                break
        model = model[:i+1]

        return model, style_losses, content_losses

    def run_style_transfer(self, num_steps=300):
        input_img = self.content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                self.model(input_img)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                loss = style_score * self.style_weight + content_score * self.content_weight
                loss.backward()

                run[0] += 1
                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)
        return input_img

def perform_style_transfer(content_image, style_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyleTransferModel(content_image, style_image, device)
    output = model.run_style_transfer()
    return output

# Authenticate and connect to Google Drive
service = authenticate_google_drive()

if service:
    folder_id = st.text_input("Enter the Google Drive folder ID:")
    if folder_id:
        images_metadata = list_images_in_folder(service, folder_id)
        
        if images_metadata:
            selected_image = st.selectbox(
                "Select an image to process", 
                [img['name'] for img in images_metadata]
            )

            if selected_image:
                img_metadata = next(img for img in images_metadata if img['name'] == selected_image)
                img = load_image_cached(service, img_metadata['id'])
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
                            generated_image = Image.open(requests.get(generated_image_url, stream=True).raw)
                            st.image(generated_image)

                            # Apply neural style transfer using PyTorch
                            style_image = load_image_cached(service, img_metadata['id'])
                            output_image_tensor = perform_style_transfer(generated_image, style_image)

                            output_image = transforms.ToPILImage()(output_image_tensor.squeeze(0))
                            st.image(output_image, caption="Image after PyTorch Style Transfer")

                            # Explanation of how images informed the new generation
                            explanation = (
                                f"The new image was generated based on a refined prompt informed by the images in the selected folder. "
                                f"Textual descriptions of the images ({description}) and the detected emotions "
                                f"({', '.join(emotions)}) were used to refine the original prompt, shaping the content and mood of the new image. "
                                f"The style of the original image was then applied to the generated image using neural style transfer with PyTorch. "
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
