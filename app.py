import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Streamlit UI
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type your symptom or disease to get medical department, causes, symptoms & precautions.")

# Load credentials from Streamlit secrets
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM watsonx.ai model setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Model parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Function to generate medical response
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

# Input box
query = st.text_input("🔍 Enter Symptom or Disease:")

# Show result
if query:
    with st.spinner("Analyzing..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

