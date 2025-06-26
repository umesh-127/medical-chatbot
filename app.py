import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from fpdf import FPDF
from datetime import datetime
from gtts import gTTS
import os

import streamlit.components.v1 as components
from streamlit_audiorec import st_audiorec
import speech_recognition as sr
from pydub import AudioSegment

# Streamlit UI
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Speak or type your symptom to get medical advice (department, causes, symptoms & precautions).")

# Load credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM watsonx.ai model setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Generate AI response
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
    return model.generate_text(prompt=prompt, params=parameters)

# Generate PDF
def generate_pdf(symptom, response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Medical Chatbot Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"User Query: {symptom}")
    pdf.ln(5)
    clean_response = response.encode('latin1', 'ignore').decode('latin1')
    pdf.multi_cell(0, 10, f"Response:\n{clean_response}")
    filename = "medical_report.pdf"
    pdf.output(filename)
    return filename

# Text input
query = st.text_input("🔍 Enter Symptom or Disease:")

# Voice input
st.markdown("### 🎙️ Or record your voice below:")
audio_data = st_audiorec()

if audio_data is not None:
    # Save voice as wav
    with open("recorded.wav", "wb") as f:
        f.write(audio_data)

    # Convert to recognisable format
    audio = AudioSegment.from_file("recorded.wav")
    audio.export("converted.wav", format="wav")

    r = sr.Recognizer()
    with sr.AudioFile("converted.wav") as source:
        audio_text = r.listen(source)
        try:
            query = r.recognize_google(audio_text)
            st.success(f"Recognized Speech: {query}")
        except:
            st.error("Sorry, could not recognize the speech.")

# Generate and show result
if query:
    with st.spinner("Analyzing..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Generate voice output
        tts = gTTS(result)
        tts.save("response.mp3")
        audio_file = open("response.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

        # PDF download
        pdf_file = generate_pdf(query, result)
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Report (PDF)", data=f, file_name=pdf_file, mime="application/pdf")
