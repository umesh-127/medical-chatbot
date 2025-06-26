import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
from fpdf import FPDF
import os
import re

# Streamlit UI setup
st.set_page_config(page_title="AI Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# IBM WatsonX credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM WatsonX model setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Generation parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Function to generate response
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
    return model.generate_text(prompt=prompt, params=parameters)

# Function to clean text for PDF (remove emojis and non-latin characters)
def clean_text_for_pdf(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# User input
query = st.text_input("🔍 Enter Symptom or Disease:")

if query:
    with st.spinner("Analyzing..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Voice output
        tts = gTTS(result, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")

        # Download as PDF
        if st.button("📄 Download Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            clean_result = clean_text_for_pdf(result)
            pdf.multi_cell(0, 10, f"AI Medical Chatbot Report\n\nSymptom/Disease: {query}\n\nMedical Guidance:\n{clean_result}")
            pdf.output("medical_report.pdf")

            with open("medical_report.pdf", "rb") as file:
                st.download_button(
                    label="📥 Click to Download PDF",
                    data=file,
                    file_name="medical_report.pdf",
                    mime="application/pdf"
                )
