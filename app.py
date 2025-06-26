import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from gtts import gTTS
import io

# UI Config
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🧠 AI Medical Chatbot")
st.write("Type or speak your symptom to receive medical guidance.")

# Secrets (make sure these are set in .streamlit/secrets.toml)
api_key = st.secrets["api_key"]
region = st.secrets["region"]
project_id = st.secrets["project_id"]

# IBM watsonx.ai setup
creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
model = Model(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)

# Generation parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300
}

# Prompt construction
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

# Input query (text or voice)
query = ""

# Optional: Try voice input if running locally
voice_input = st.button("🎙️ Speak your symptom (works locally only)")
try:
    import speech_recognition as sr
    if voice_input:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
except:
    st.warning("🎤 Voice input only works when running on your computer (not in browser/cloud).")

# Or manual input
query = st.text_input("✍️ Or type your symptom or disease:", value=query)

# Language selection for output voice
lang_choice = st.selectbox(
    "🌐 Select language for voice response:",
    options=["en", "hi", "te"],
    format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x]
)

# Show medical guidance
if query:
    with st.spinner("Analyzing with IBM watsonx..."):
        result = get_medical_response(query)
        st.markdown("### 🧾 Medical Guidance")
        st.markdown(result)

        # Convert result to voice using gTTS
        try:
            tts = gTTS(text=result, lang=lang_choice)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Audio generation failed: {e}")
