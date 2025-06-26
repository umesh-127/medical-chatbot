import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from googletrans import Translator
from fpdf import FPDF
from datetime import datetime

# ------------------ Streamlit Config ------------------ #
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# ------------------ Load Secrets ------------------ #
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# ------------------ IBM Model Setup ------------------ #
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# ------------------ Keywords for Emergency ------------------ #
emergency_keywords = ["chest pain", "shortness of breath", "severe bleeding", "heart attack", "unconscious", "stroke"]

# ------------------ Functions ------------------ #
def detect_emergency(text):
    for keyword in emergency_keywords:
        if keyword.lower() in text.lower():
            return True
    return False

def translate_to_english(text):
    translator = Translator()
    result = translator.translate(text, dest='en')
    return result.text

def generate_pdf(symptom, response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="🧠 AI Medical Chatbot Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"User Query: {symptom}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Response:\n{response}")
    filename = "medical_report.pdf"
    pdf.output(filename)
    return filename

def get_medical_response(symptom):
    prompt = f"""A patient says: "{symptom}"

Based on this, provide:
1. Medical department to consult.
2. Possible related causes.
3. Common symptoms.
4. Safe precautions or home remedies.
5. Add a note advising the user to consult a doctor.
6. (Optional) Suggest if appointment with doctor is needed.

Format the response clearly as bullet points.
"""
    return model.generate_text(prompt=prompt, params=parameters)

# ------------------ Input UI ------------------ #
query = st.text_input("🔍 Enter Symptom or Disease (any language):")

if query:
    with st.spinner("Analyzing..."):

        english_query = translate_to_english(query)
        emergency = detect_emergency(english_query)
        result = get_medical_response(english_query)

        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        if emergency:
            st.error("🚨 *Warning: Your symptoms may indicate a medical emergency. Please contact your local emergency services immediately.*")

        st.info("📅 *If symptoms persist, consider booking an appointment with a certified doctor near you.*")

        pdf_file = generate_pdf(query, result)
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Report (PDF)", data=f, file_name=pdf_file, mime="application/pdf")
