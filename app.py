import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
from fpdf import FPDF
import qrcode
from PIL import Image
import os
import re
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
from datetime import datetime, timedelta

# --- Streamlit Page Settings ---
st.set_page_config(page_title="🧠 AI Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# --- Load IBM WatsonX Credentials ---
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# --- IBM WatsonX Model Setup ---
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# --- Generation Parameters ---
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# --- Function to Generate Medical Guidance ---
def get_medical_response(symptom):
    prompt = f'''A patient says: "{symptom}".

Based on this, provide:
1. Medical department to consult.
2. Possible related causes.
3. Common symptoms.
4. Safe precautions or home remedies.
5. Add a note advising the user to consult a doctor.
Also mention a confidence score between 85% to 100%.

Format clearly using bullet points.
'''
    return model.generate_text(prompt=prompt, params=parameters)

# --- Clean text for PDF ---
def clean_text_for_pdf(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# --- Generate QR code ---
def generate_qr(text, filename="qr_code.png"):
    qr = qrcode.make(text)
    qr.save(filename)
    return filename

# --- Send Email with Attachment ---
def send_email_with_pdf(receiver_email, pdf_path):
    sender_email = st.secrets["sender_email"]
    sender_password = st.secrets["sender_password"]

    msg = EmailMessage()
    msg['Subject'] = 'Your AI Medical Chatbot Report'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content("Please find attached your AI-generated medical report.")

    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
        msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename="medical_report.pdf")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        return str(e)

# --- Send Feedback to Owner ---
def send_feedback_email(feedback):
    try:
        sender_email = st.secrets["sender_email"]
        sender_password = st.secrets["sender_password"]
        owner_email = st.secrets["owner_email"]

        msg = EmailMessage()
        msg["Subject"] = "User Feedback from Medical Chatbot"
        msg["From"] = sender_email
        msg["To"] = owner_email
        msg.set_content(f"Feedback received: {feedback}")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Error sending feedback email: {e}")

# --- Upload Medical Image ---
st.markdown("### 🖼️ Upload Medical Image (optional)")
image_file = st.file_uploader("Upload an X-ray, MRI, or other medical image:", type=["png", "jpg", "jpeg"])
if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

# --- User Input ---
query = st.text_input("🔍 Enter Symptom or Disease:")

if query:
    with st.spinner("Analyzing..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # --- Voice Output ---
        tts = gTTS(result, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")

        # --- PDF Generation ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"AI Medical Chatbot Report\n\nSymptom/Disease: {query}\n\nMedical Guidance:\n{clean_text_for_pdf(result)}")
        pdf_path = "medical_report.pdf"
        pdf.output(pdf_path)

        st.download_button("📄 Download PDF Report", data=open(pdf_path, "rb"), file_name="medical_report.pdf", mime="application/pdf")

        # --- QR Code for Result ---
        qr_path = generate_qr(query + "\n\n" + clean_text_for_pdf(result))
        st.image(qr_path, caption="📷 Scan QR to View Summary")

        # --- Email Feature ---
        with st.expander("📧 Send Report via Email"):
            email_input = st.text_input("Enter recipient email:")
            if st.button("Send Email"):
                try:
                    valid = validate_email(email_input)
                    email = valid.email
                    status = send_email_with_pdf(email, pdf_path)
                    if status is True:
                        st.success("✅ Email sent successfully.")
                    else:
                        st.error(f"❌ Failed to send email: {status}")
                except EmailNotValidError as e:
                    st.error(f"Invalid Email: {e}")

        # --- Feedback System ---
        st.markdown("### 🙋 Was this guidance helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
                send_feedback_email("👍 Helpful")
        with col2:
            if st.button("👎 No"):
                st.warning("We'll keep improving. Thanks!")
                send_feedback_email("👎 Not helpful")

        # --- Reminder & Appointment Suggestion ---
        st.markdown("### 📅 Want to schedule a doctor visit?")
        appointment_time = st.date_input("Choose an appointment date:", datetime.now() + timedelta(days=1))
        if st.button("📤 Email Appointment Reminder"):
            if email_input:
                msg = EmailMessage()
                msg['Subject'] = 'Doctor Appointment Reminder'
                msg['From'] = st.secrets["sender_email"]
                msg['To'] = email_input
                msg.set_content(f"Your doctor appointment is scheduled for {appointment_time.strftime('%Y-%m-%d')}. Kindly keep a note of it.")
                try:
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(st.secrets["sender_email"], st.secrets["sender_password"])
                        smtp.send_message(msg)
                    st.success("📧 Reminder email sent!")
                except Exception as e:
                    st.error(f"Error sending reminder: {e}")

# --- Footer Disclaimer ---
st.markdown("---")
st.markdown("*Disclaimer: This is AI-generated guidance and should not replace consultation with a licensed healthcare provider.*")
