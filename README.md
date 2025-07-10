🌿 SoulCare – Mental Healthcare Web App
SoulCare is an AI-powered mental healthcare web application designed to provide empathetic, private, and accessible emotional support through conversational AI, mood tracking, and self-care resources. Built with modern web technologies and NLP, it serves as a first-level support system for individuals experiencing emotional distress, especially anxiety and depression.

🧠 Features
🤖 AI Chatbot (LLaMA-based) – Engages users in human-like conversations using NLP and sentiment analysis.

📊 Mood Tracker – Logs daily emotions and visualizes trends over time.

🧘 Guided Meditation – Offers curated relaxation techniques based on the user’s emotional state.

🆘 Emergency Support – Suggests helpline numbers when signs of crisis are detected.

📈 Analytics Dashboard – Provides visual insights into the user's emotional journey.

🔒 Secure Authentication – User registration/login using JWT with encrypted data storage.

🚀 Tech Stack
Layer	Technologies
Frontend	React.js, Tailwind CSS
Backend	Node.js, Express.js
Database	MongoDB Atlas
AI/NLP	Python, Hugging Face Transformers, NLTK, TensorFlow
llama

📐 System Architecture
Presentation Layer: Chatbot UI, mood log, dashboard (React.js)

Application Layer: Authentication, routing, and APIs (Node.js, Express)

Database Layer: Stores user data, logs, and chatbot history (MongoDB)

AI Layer: Sentiment analysis and intent recognition (Python NLP pipeline)

🛠️ Installation Guide
Prerequisites
Node.js and npm

Python 3.8+

MongoDB Atlas account

Hugging Face Transformers

Frontend Setup
bash
Copy
Edit
cd client
npm install
npm run dev
Backend Setup
bash
Copy
Edit
cd server
npm install
node index.js
Python NLP Service
bash
Copy
Edit
cd ai
pip install -r requirements.txt
python app.py
📷 Screenshots
Include the following images in your repo:

Login Page

Mood Tracker Interface

Chatbot Conversation View

Meditation Library

Dashboard Analytics

📊 Results & Testing
Intent Accuracy: 84%

Sentiment Detection Accuracy: 88%

Avg Response Time: ~1.2 seconds

User Satisfaction: 92% rated UI as “Excellent” or “Good”

🔐 Ethical Considerations
User data is stored with explicit consent and encrypted storage.

AI does not provide clinical advice.

All high-risk inputs are flagged with emergency suggestions.

Future versions aim to support multilingual and therapist integration.

📈 Future Enhancements
📱 Mobile App (Android/iOS)

🗣️ emotion detection

🧑‍⚕️ Therapist live support portal

🌐 Multilingual NLP (Hindi, Tamil, Bengali)

⚠️ Real-time crisis alert system

📄 License
This project is built for academic and research purposes. Contact the authors for extended use or publication.

👨‍💻 Developed By
Pushp Choudhary

Institution: Department of Computer Science, NIET, Greater Noida
