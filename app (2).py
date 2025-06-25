
import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Page setup
st.set_page_config(page_title="AI Medical Chatbot", layout="centered")
st.title("🩺 AI Medical Chatbot - Symptom Guidance")

# IBM watsonx credentials
api_key = "xgu0BhaIxpPcwrb2N4vD13gu7RulnBw95CcAJN4_K_Yd"
region = "us-south"
project_id = "797a0add-2e8f-45ed-80dc-8cf27f0541b9"

# Connect to IBM watsonx.ai
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Function to get response
def recommend_info(symptoms):
    prompt = f"""A patient reports: '{symptoms}'

Based on this, provide:
1. Which medical department should they consult.
2. What could be the possible causes or related conditions.
3. What are the typical symptoms to watch for.
4. What basic precautions or home remedies can be taken (if safe).
5. Add a note reminding the user to consult a doctor for diagnosis.

Format the response clearly in bullet points.""" 
    response = model.generate_text(prompt=prompt, params=parameters)
    return response

# User input
symptoms = st.text_input("Enter your symptoms 👇")

if symptoms:
    with st.spinner("Analyzing your symptoms..."):
        answer = recommend_info(symptoms)
        st.markdown("### 🧠 Health Guidance")
        st.markdown(answer)
