# 📚 Chat with Multiple PDFs (Offline AI)
An Offline AI PDF Question Answering System built using:

🧠 Ollama (Local LLM – phi3)
🔎 FAISS (Vector Database)
🔗 LangChain (Retrieval Pipeline)
📄 Sentence Transformers (Embeddings)
🌐 Streamlit (User Interface)

This project allows users to upload PDFs and ask questions about them — completely offline, without any API keys.

🚀 Features
✅ Fully Offline (No Internet Required After Setup)

✅ No API Key Needed

✅ Optimized for 8GB RAM Laptops

✅ Fast Retrieval-Based Answers

✅ Lightweight Local LLM (phi3:mini)

✅ Clean Streamlit UI

🛠️ System Requirements
• Windows / Mac / Linux  
• Python 3.10+  
• 8GB RAM Recommended  
• Ollama Installed  

📦 Installation Guide
Step 1 — Clone Repository
 git clone https://github.com/Nidhisharora/ask-multiple-pdfs.git

 cd ask-multiple-pdfs

 Install Python 3.10 or 3.11 only

Step 2 — Create Virtual Environment
python -m venv venv
venv\Scripts\activate


(Mac/Linux)
source venv/bin/activate

Step 3 — Install Dependencies
pip install -r requirements.txt

Step 4 — Install Ollama
Download from:
https://ollama.com

Verify installation:
ollama --version

Step 5 — Pull Required Model
ollama pull phi3
You can check installed models:
ollama list

▶️ Running the Application
Start Ollama (if not running):
   ollama serve
Then run Streamlit:
   streamlit run app.py

📖 How to Use
1. Upload one or more PDFs.
2. Click **Process**.
3. Ask questions about your documents.
4. Get answers generated using local AI.
5. Use "Clear Chat" if needed.


🧠 Architecture

PDF → Text Extraction → Text Chunking →  
Embeddings → FAISS Vector Store →  
Retriever → Local LLM (phi3:mini) → Answer

🏆 Why This Project?
• Works fully offline  
• No OpenAI API cost  
• Privacy safe  
• Hackathon friendly  
• Lightweight and deployable  

📌 Future Improvements
• Add PDF highlighting
• Add chat streaming response
• Add multi-user support
• Add better UI themes
• Deploy on local server machine

👨‍💻 Author
Nidhish Arora
GitHub:
https://github.com/Nidhisharora

⭐ If you like this project, give it a star!
