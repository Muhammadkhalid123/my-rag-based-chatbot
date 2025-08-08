from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import re
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# === Prompt Template ===
custom_prompt = PromptTemplate(
    template="""
You are an assistant for KDP Digital Publishers.
Answer the question using only the provided context.
Be clear, concise, and human in tone.
Do not say things like "according to the provided text" or "based on the context".
If the answer is not in the context, say you don’t know.
Keep your answer short and to the point.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# === Lead Capture Helpers ===
def is_lead_capture_query(query: str) -> bool:
    return any(keyword in query.lower() for keyword in [
        "talk to someone", "connect", "call", "speak with", "contact",
        "reach out", "communication", "on phone", "get in touch", "someone to talk"
    ])

def is_valid_email(email: str) -> bool:
    return "@" in email and "." in email

def is_valid_phone(phone: str) -> bool:
    phone_digits = re.sub(r"[^\d]", "", phone)
    return len(phone_digits) >= 6

def save_lead_to_json(name: str, email: str, phone: str):
    lead_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "timestamp": datetime.now().isoformat()
    }
    file_path = "lead_capture.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(lead_data)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

# === Load Knowledge Base ===
pdf_path = "data/KDP_Digital_Publishers_Website_Information.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# === RAG QA Chain ===
llm = OllamaLLM(model="llama3")  # change if using another model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# === Endpoints ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Lead capture intent
    if is_lead_capture_query(question):
        return jsonify({
            "lead_required": True,
            "message": "I can connect you with our team. Please provide your name, email, and phone."
        })

    # Run through RAG
    result = qa_chain.run(question)
    return jsonify({
        "lead_required": False,
        "answer": result
    })

@app.route("/lead", methods=["POST"])
def lead():
    data = request.get_json()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    errors = []
    if not name:
        errors.append("Name is required.")
    if not is_valid_email(email):
        errors.append("Valid email is required.")
    if not is_valid_phone(phone):
        errors.append("Valid phone number is required.")

    if errors:
        return jsonify({"errors": errors}), 400

    save_lead_to_json(name, email, phone)
    return jsonify({"message": "Lead saved successfully"})

if __name__ == "__main__":
    app.run(debug=True)
