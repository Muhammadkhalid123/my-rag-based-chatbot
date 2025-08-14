from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import re
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
CORS(app)

# -------------------------
# Load PDF and create vectorstore - KEEP ORIGINAL
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "vector_store/kdp_faiss_index"
PDF_PATHS = [
    "data/Kinetic Digital Publishers (KDP).pdf"
]

def initialize_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        print("🔍 Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("📚 Creating new vector store...")
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for pdf_path in PDF_PATHS:
            if os.path.exists(pdf_path):
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                docs = text_splitter.split_documents(documents)
                all_docs.extend(docs)
                print(f"✅ Loaded {len(docs)} chunks from {pdf_path}")
            else:
                print(f"⚠️ PDF not found: {pdf_path}")
        
        if all_docs:
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            vectorstore.save_local(VECTOR_STORE_PATH)
            print(f"💾 Vector store saved with {len(all_docs)} total chunks")
            return vectorstore
        else:
            raise Exception("No documents found to create vector store")

vectorstore = initialize_vector_store()

# SIMPLE - Just one retriever like your original
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM setup - KEEP YOUR ORIGINAL
# -------------------------
llm = OllamaLLM(
    model="gemma3:27b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://192.168.18.10:11434")
)


# -------------------------
# SIMPLE prompt like your original - NOT COMPLEX
# -------------------------
conversation_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template="""You are a helpful assistant for KDP Digital Publishers.

Rules for answering:
1. If the user says "you" or "you guys" it means KDP Digital Publishers.
2. Always answer clearly, concisely, and with a human tone. Never use asterisks (*), use dashes (-) or numbers instead.
3. If the question is about packages:
   - Analyze the user's previous responses in the conversation to understand their needs
   - Show the most relevant packages first, or filter packages based on their requirements
   - List package names with prices, one per line, with brief relevant descriptions
   - After listing, leave two blank lines and ask: "Which package are you most interested in?"
4. If the question is about services:
   - First analyze what specific type of service the user is asking about from their question
   - If they mention a specific service category (like "publishing", "marketing", "design"), filter and show only relevant services from the context
   - If they ask generally about "services", show all available services
   - Extract only the short service names (one per line, no descriptions)
   - Ensure each service name starts with a dash and a space for consistent parsing
   - After listing, leave two blank lines and ask: "Which service are you most interested in?"
5. If the user directly mentions a service (e.g., "I want to publish my book", "I need cover design", "I want ghost writing"):
   - Do not list all services.
   - Treat this as a confirmed selection of that service.
   - First, acknowledge their interest warmly (e.g., "That’s great! We’d be happy to help you with [service].")
   - Ask one or two gentle, service-specific questions to better understand their requirements.
       Example: For "publish my book", ask: "Do you already have your manuscript ready, or are you still working on it?"
   - Once they respond, naturally introduce the available plans/packages that fit their answers.
       Example: "Based on what you’ve told me, here are the publishing plans we can offer..."
6. If the user asks "what is [service]" or "what does [service] include" or similar informational questions:
   - Provide a clear, concise explanation of what that specific service includes
   - Mention the key benefits and what's typically involved
   - After explaining, ask: "Does this sound like something that would help with your project?"
   - Do not immediately jump to qualification questions
7. If the user directly mentions a service (e.g., "I want to publish my book", "I need cover design", "I want ghost writing"):
   - Do not list all services.
   - Treat this as a confirmed selection of that service.
   [rest of existing rule...]
8. If the user selects or mentions a specific service from a previous list:
   - Do not immediately list packages.
   - First, acknowledge their interest warmly (e.g., "Great! We'd be happy to help you with [service].")
   - Ask one or two gentle, service-specific questions to better understand their requirements.
       Example: For "formatting", ask: "What type of document do you need formatted - manuscript, ebook, or print book?"
   - Once they respond, naturally introduce the available packages that fit their needs.
       Example: "Based on what you've told me, here are the formatting packages we offer..."
   - Follow package listing rules (point 3).
9. If the user selects a package:
   - This should trigger the lead capture form automatically.
   - Do not ask for contact details in text format.
10. When a user responds to clarifying questions about their service needs:
   - Analyze their response and match it to the most relevant packages from the context
   - If their needs clearly align with specific packages, highlight those first or show only the most relevant ones
   - If they mention "content", "structure", "plot", or "development" for editing, focus on comprehensive/advanced packages
   - If they mention "basic", "simple", or "quick" focus on basic packages
   - Explain briefly why you're recommending specific packages based on their stated needs
   - After showing relevant packages, ask: "Which of these packages seems like the best fit for your needs?"
11. If the user greets you (hi, hello), reply politely without contact details unless they ask.
12. Provide contact details only if directly asked for them.
13. Never add "according to the context" or similar filler.
14. No extra blank lines except where specified.
15. Always guide the conversation forward — never repeat the same question if it has already been answered.
16. If the user is exploring and not ready to commit, give concise helpful information and ask the most relevant next question.
17. When you are giving one package to you then do not say "which packages are you interested in" just say "Are you interested in this package"

Context: {context}

Previous conversation: {history}

Question: {question}

Response:"""
)

# Store conversation history
chat_sessions = defaultdict(list)

# -------------------------
# SIMPLE intent detection - not complex
# -------------------------
def detect_intent(question, history=""):
    """Simple intent detection"""
    q_lower = question.lower()
    
    # Informational questions about services
    if any(phrase in q_lower for phrase in ["what is", "what does", "tell me about", "explain"]):
        return "service_info"
    
    # Package selection signals
    package_selection = [
        "the one you are suggesting", "that one", "the classic plan", "basic plan", 
        "premium plan", "standard plan", "i'll take", "i want that", "sounds good"
    ]
    
    # Real buy signals
    buy_signals = ["proceed", "ready to start", "let's go", "sign up", "get started"]
    
    if any(signal in q_lower for signal in package_selection + buy_signals):
        return "buy"
    
    return "general"
# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return send_from_directory('frontend', 'index.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")
    
    if not question:
        return jsonify({"error": "No message provided"}), 400

    # Get session history
    session_history = chat_sessions[session_id]
    
    # Simple history text (last 3 exchanges only)
    history_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" 
                              for msg in session_history[-3:]])
    
    # Simple intent detection
    intent = detect_intent(question, history_text)
    
    # Handle buy intent immediately
    if intent == "buy":
        answer = "Great! To get you connected with our publishing team, I'll need your contact information. Please provide your name, email, and phone number."
        
        session_history.append({
            "user": question,
            "bot": answer
        })
        
        return jsonify({
            "lead_required": True,
            "answer": answer,
            "session_id": session_id
        })
    
    # Get context - SIMPLE, not multiple searches
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs[:2]])  # Only 2 docs
        context = context[:1000]  # Limit context size
    except Exception as e:
        context = ""
    
    # Build simple prompt
    prompt = conversation_prompt.format(
        context=context,
        history=history_text,
        question=question
    )
    
    try:
        # Get LLM response
        answer = llm.invoke(prompt)
        answer = answer.strip()
        
    except Exception as e:
        answer = "I'm here to help with your publishing needs. What would you like to know?"
    
    # Save to session history
    session_history.append({
        "user": question,
        "bot": answer
    })
    
    # Keep history small
    if len(session_history) > 10:
        session_history.pop(0)
    
    return jsonify({
        "lead_required": False,
        "answer": answer,
        "session_id": session_id
    })

# -------------------------
# Lead capture endpoint
# -------------------------
@app.route("/lead", methods=["POST"])
def lead():
    data = request.get_json()
    if not data:
        return jsonify({"errors": ["No data provided"]}), 400

    # Simple validation
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    best_time = data.get("best_time", "").strip()
    session_id = data.get("session_id", "default")

    errors = []
    if not name:
        errors.append("Name is required")
    
    if not email and not phone:
        errors.append("Email or phone is required")

    if errors:
        return jsonify({"errors": errors}), 400

    try:
        # Save lead
        lead_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "best_time": best_time,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "brand": "KDP Digital Publishers"
        }
        
        save_lead_to_json(lead_data)
        
        # Clear session
        if session_id in chat_sessions:
            chat_sessions[session_id].clear()
        
        return jsonify({
            "success": True,
            "message": "Thank you! Your information has been received.",
            "next_steps": "Our team will contact you within 24 hours."
        })
        
    except Exception as e:
        return jsonify({"errors": ["Failed to save. Please try again."]}), 500

# -------------------------
# Simple lead saving function
# -------------------------
def save_lead_to_json(lead_data):
    """Save lead data to JSON file"""
    file_path = "leads/lead_capture.json"
    os.makedirs("leads", exist_ok=True)
    
    try:
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if content:
                    existing_data = json.loads(content)
        
        existing_data.append(lead_data)
        
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        raise

# -------------------------
# Debug endpoints
# -------------------------
@app.route("/test-llm", methods=["GET"])
def test_llm():
    try:
        response = llm.invoke("Say hello")
        return jsonify({"status": "success", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/clear_session", methods=["POST"])
def clear_session():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    
    if session_id in chat_sessions:
        chat_sessions[session_id].clear()
    
    return jsonify({"message": "Session cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
