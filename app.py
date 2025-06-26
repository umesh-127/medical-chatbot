import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
from fpdf import FPDF
import os

# Streamlit UI setup
st.set_page_config(page_title="🧠 AI Medical Chatbot", layout="centered")
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

# Medical Response Generator
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

# User Input
query = st.text_input("🔍 Enter Symptom or Disease:")

# When input is provided
if query:
    with st.spinner("Generating medical guidance..."):
        result = get_medical_response(query)

        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Text-to-speech
        tts = gTTS(result, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")

        # PDF Download Button
        if st.button("📄 Download Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"🧠 AI Medical Chatbot Report\n\nSymptom/Disease: {query}\n\nMedical Guidance:\n{result}")
            pdf.output("medical_report.pdf")

            with open("medical_report.pdf", "rb") as file:
                st.download_button(
                    label="📥 Click to Download PDF",
                    data=file,
                    file_name="medical_report.pdf",
                    mime="application/pdf"
                )
