import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from fpdf import FPDF
from datetime import datetime
from gtts import gTTS
import os

# Streamlit UI
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# Load credentials from secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM Watsonx setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Generation params
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Get AI response
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

# PDF export
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

# Input
query = st.text_input("🔍 Enter Symptom or Disease:")

if query:
    with st.spinner("Analyzing..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Voice output
        tts = gTTS(result)
        tts.save("response.mp3")
        with open("response.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        # PDF download
        pdf_file = generate_pdf(query, result)
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Report (PDF)", data=f, file_name=pdf_file, mime="application/pdf")
