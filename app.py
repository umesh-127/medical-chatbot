import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
import speech_recognition as sr
import os

# Page config
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# Load IBM credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# Setup IBM Watsonx model
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Define function to get model response
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
query = ""

# Button for voice input
if st.button("🎙️ Speak Symptom"):
    with sr.Microphone() as source:
        st.info("🎧 Listening for your voice...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            query = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {query}")
        except sr.UnknownValueError:
            st.error("Sorry, could not understand your voice.")
        except sr.RequestError:
            st.error("Speech Recognition service failed.")

# Manual text input
query = st.text_input("🔍 Enter Symptom or Disease:", value=query)

# Language selection for voice output
lang_choice = st.selectbox("🌐 Choose Voice Language:", ["en", "hi", "te"], format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x])

# Generate response and play TTS
if query:
    with st.spinner("🧠 Analyzing your input..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Convert text to speech
        tts = gTTS(text=result, lang=lang_choice)
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")
