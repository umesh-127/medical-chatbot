import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
import speech_recognition as sr
import os
from io import BytesIO
from fpdf import FPDF

# Streamlit UI setup
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Speak or type your symptom/disease to get medical advice.")

# IBM watsonx credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM watsonx model setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Model generation parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Function to generate AI medical response
def get_medical_response(symptom):
    prompt = f"""A patient says: "{symptom}"

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

# Voice input logic
recognizer = sr.Recognizer()
query = ""

if st.button("🎙️ Speak your symptom"):
    with sr.Microphone() as source:
        st.info("🎧 Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            query = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {query}")
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("❌ Speech recognition service failed.")

# Manual input fallback
query = st.text_input("✍️ Or type your symptom:", value=query)

# Language selection
lang_choice = st.selectbox(
    "🌐 Select language for audio response:",
    ["en", "hi", "te"],
    format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x]
)

# Cleanup old audio
if os.path.exists("response.mp3"):
    os.remove("response.mp3")

# Handle response
if query:
    with st.spinner("🧠 Analyzing your symptom..."):
        result = get_medical_response(query)

    if result.strip():
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Text-to-speech
        try:
            tts = gTTS(result, lang=lang_choice)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")
        except Exception as e:
            st.error(f"⚠️ Text-to-speech failed: {e}")

        # Optional PDF generation
        if st.button("📄 Download as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in result.split('\n'):
                pdf.multi_cell(0, 10, line)
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button(label="📥 Click to Download", data=pdf_output.getvalue(),
                               file_name="medical_guidance.pdf", mime="application/pdf")
    else:
        st.warning("The model returned an empty response. Try rephrasing your query.")
