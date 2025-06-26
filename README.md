# 🧠 AI Medical Chatbot

An AI-powered chatbot built using **IBM WatsonX Granite Model** that helps users get medical department suggestions, causes, symptoms, precautions, and confidence scores based on the entered disease or symptom. The app also supports advanced features like voice response, PDF report, email delivery, QR code summary, image upload, feedback system, and appointment reminders.

---

## 🚀 Features

- 💬 Symptom-to-guidance using IBM watsonx.ai foundation model
- 🔊 Voice output using Google Text-to-Speech (gTTS)
- 📄 PDF report generation with guidance
- 📧 Email the report to any user
- 🖼️ Upload X-ray/MRI image support
- 📷 Auto-generated QR code for guidance summary
- 🗓️ Appointment reminder via email
- 👍👎 Feedback with email to owner

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt contains:

nginx
Copy
Edit
streamlit
ibm-watsonx-ai
gTTS
fpdf
qrcode
Pillow
email-validator
🔐 Add Secret Credentials
Create a file at:

bash
Copy
Edit
.streamlit/secrets.toml
Paste the following:

toml
Copy
Edit
api_key = "YOUR_IBM_API_KEY"
region = "us-south"
project_id = "YOUR_WATSONX_PROJECT_ID"

sender_email = "your_email@gmail.com"
sender_password = "your_gmail_app_password"
owner_email = "your_email@gmail.com"
⚠️ Use a Gmail App Password instead of your actual Gmail password.

▶️ Run the App Locally
bash
Copy
Edit
streamlit run app.py
Then go to: http://localhost:8501

📷 Sample Screenshot

🧪 Sample Usage
Input:

nginx
Copy
Edit
headache
Response Includes:

Department: Neurology / General Medicine

Causes: Migraine, Tension Headache, Sinusitis

Symptoms: Pain, Nausea, Photophobia

Precautions: Rest, Hydration, Painkillers

Voice playback

PDF download

QR code summary

Email delivery

Feedback system

Appointment email reminder

📁 File Structure
markdown
Copy
Edit
medical-chatbot/
├── app.py
├── requirements.txt
└── .streamlit/
    └── secrets.toml
☁️ Deploy on Streamlit Cloud
Push to GitHub

Go to streamlit.io/cloud

Link your GitHub repo

Add your secret keys in the dashboard

Click Deploy

📢 Feedback System
👍 "Yes" → Owner receives email: Helpful

👎 "No" → Owner receives email: Not Helpful

🗓️ Appointment Reminder
Users pick a date from the calendar

Email sent to the user confirming the appointment reminder

🔒 Security Notes
Add .streamlit/secrets.toml to .gitignore

Never expose your credentials publicly

Use Gmail App Passwords only

🧠 Built With
IBM WatsonX AI

Streamlit

Google TTS

FPDF

qrcode

Pillow

email-validator

📜 License
This project is licensed under the MIT License.
Free to use and modify with attribution.

🙌 Support
If you like this project, leave a ⭐ on GitHub.
For issues, open a ticket or message the owner via email.

yaml
Copy
Edit

---

You can copy and paste this directly into your `README.md` file on GitHub.  
Let me know if you'd also like a `sample_screenshot.png` image or a deploy badge!







