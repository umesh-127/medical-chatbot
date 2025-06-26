import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
import speech_recognition as sr
import os

# Streamlit Page Setup
st.set_page_config(page_title="🧠 AI Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type or speak your symptom/disease to get medical department, causes, symptoms & precautions.")

# Load IBM credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM watsonx.ai setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Model generation parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Generate AI response
def get_medical_response(symptom):
    prompt = f"""A patient says: \"{symptom}\"

Based on this, provide:
1. Medical department to consult.
2. Possible related causes.
3. Common symptoms.
4. Safe precautions or home remedies.
5. Add a note advising the user to consult a doctor.

Format the response clearly as bullet points.
"""
    response = model.generate_text(prompt=prompt, params=parameters)
    return response

# Voice recognition
recognizer = sr.Recognizer()
voice_input = st.button("🎙️ Speak your symptom")
query = ""

if voice_input:
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service error.")

# Manual text input
query = st.text_input("🔍 Enter Symptom or Disease:", value=query)

# Language selector (Always visible)
lang_choice = st.selectbox(
    "🌐 Choose Voice Language:",
    ["en", "hi", "te"],
    format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x]
)

# Process input and show response
if query:
    with st.spinner("Analyzing with IBM watsonx AI..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Text to speech in selected language
        tts = gTTS(result, lang=lang_choice)
        tts.save("response.mp3")
        audio_file = open("response.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
