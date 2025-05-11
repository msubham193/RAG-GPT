import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field, conint
from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import PyPDF2
import tempfile
import shutil
import uuid
import pymongo
import hashlib
from datetime import datetime, timedelta
from bson import ObjectId
import threading
import time
import schedule

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize FastAPI app
app = FastAPI(title="CUTM Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global vector store
vector_store = None

# MongoDB Connection
MONGO_URI = "mongodb+srv://audobookcmp:mERonz15DT220mZ3@audobookserverlessinsta.oa4gzac.mongodb.net/cime?retryWrites=true&w=majority&appName=audobookServerlessInstance0"
client = pymongo.MongoClient(MONGO_URI)
db = client.cime
users_collection = db.users
sample_questions_collection = db.sample_questions
user_feedback_collection = db.user_feedback

# Models
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserInDB(UserBase):
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None

class PDFUploadResponse(BaseModel):
    message: str
    document_id: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# New models for sample questions
class SampleQuestionCreate(BaseModel):
    question: str

class SampleQuestionResponse(BaseModel):
    id: str
    question: str
    created_at: datetime

    class Config:
        from_attributes = True

# User feedback models
class UserFeedbackCreate(BaseModel):
    rating: conint(ge=1, le=5)  # Rating between 1 and 5
    comment: Optional[str] = None

class UserFeedbackResponse(BaseModel):
    id: str
    user_id: str
    rating: int
    comment: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Helper functions
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return hash_password(plain_password) == hashed_password

def read_text_file(file_path):
    """Try different encodings to read the text file."""
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise RuntimeError(f"Could not read file with any of the following encodings: {encodings}")

def extract_text_from_pdf(pdf_file_path):
    """Extract text content from a PDF file."""
    try:
        text = ""
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise

def initialize_vector_store():
    """Initialize or load the vector store."""
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Check if saved index exists
        if os.path.exists("faiss_index"):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.load_local(
                "faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector store from disk")
            return vector_store
            
        # If no saved index, create new one
        documents = []
        
        # Check if original data file exists
        if os.path.exists("data.txt"):
            text_content = read_text_file("data.txt")
            documents.append(Document(page_content=text_content, metadata={"source": "data.txt"}))
        
        # Add all documents from data directory
        for filename in os.listdir("data"):
            if filename.endswith(".txt"):
                file_path = os.path.join("data", filename)
                text_content = read_text_file(file_path)
                documents.append(Document(page_content=text_content, metadata={"source": filename}))
        
        if not documents:
            print("No documents found to index")
            return None
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(split_documents, embeddings)
        
        # Save the index to disk
        vector_store.save_local("faiss_index")
        print("Created and saved new vector store to disk")
        
        return vector_store
    
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize vector store")

def add_document_to_vector_store(text_content, document_id):
    """Add a new document to the vector store."""
    global vector_store
    
    try:
        doc = Document(page_content=text_content, metadata={"source": f"{document_id}.txt"})
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents([doc])
        
        # Save document text to file
        with open(os.path.join("data", f"{document_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # If vector store exists, add documents to it
        if vector_store:
            vector_store.add_documents(documents)
        else:
            # Create new vector store
            vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save the updated index to disk
        vector_store.save_local("faiss_index")
        print(f"Added document {document_id} to vector store")
        
        return True
    
    except Exception as e:
        print(f"Error adding document to vector store: {str(e)}")
        raise

# Initialize the chat chain
def get_chat_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='llama-3.1-8b-instant'
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant with expertise in College of IT & Management Education
Bhubaneswar called CIME . Your knowledge comes from the College official website data. Please:
    
    1. Provide accurate, factual information from CIME's website data
    2. Say "I don't have that information in my database" if you cannot find a reliable answer
    3. Keep responses clear and professional
    4. Stay focused on answering the specific question asked
    5. Use natural, conversational language while maintaining professionalism"
    6.if any question contains the images url please provide also and the respective page link.

    always give poin to point answer and if the question is not related to CIME then say "I don't have that information in my database".\
    dont give that according to website always give the answer according to the question asked.    
    
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

def delete_inactive_users():
    """Delete users who haven't been active for 7 days."""
    try:
        # Get the date 7 days ago
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        # Delete users created more than 7 days ago
        result = users_collection.delete_many({"created_at": {"$lt": seven_days_ago}})
        
        # Also delete their feedback
        user_ids = [user["_id"] for user in users_collection.find({"created_at": {"$lt": seven_days_ago}})]
        for user_id in user_ids:
            user_feedback_collection.delete_many({"user_id": str(user_id)})
        
        print(f"Deleted {result.deleted_count} inactive users and their feedback")
    except Exception as e:
        print(f"Error deleting inactive users: {str(e)}")

def schedule_user_cleanup():
    """Schedule the user cleanup task to run daily."""
    schedule.every(24).hours.do(delete_inactive_users)
    
    # Run the scheduling loop in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Sleep for 1 hour between checks
    
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

@app.on_event("startup")
async def startup_event():
    """Initialize vector store and start cleanup scheduler on startup."""
    global vector_store
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # Create PDFs directory if it doesn't exist
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
        
    vector_store = initialize_vector_store()
    
    # Start the user cleanup scheduler
    schedule_user_cleanup()

# Simple User Authentication Endpoints
@app.post("/signup", response_model=UserResponse)
async def signup(user_data: UserCreate):
    """Register a new user."""
    # Check if email already exists
    if users_collection.find_one({"email": user_data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_dict = {
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hash_password(user_data.password),
        "created_at": datetime.utcnow()
    }
    
    result = users_collection.insert_one(user_dict)
    
    # Return the created user
    created_user = users_collection.find_one({"_id": result.inserted_id})
    return {
        "id": str(created_user["_id"]),
        "email": created_user["email"],
        "name": created_user["name"],
        "created_at": created_user["created_at"]
    }

@app.post("/login", response_model=UserResponse)
async def login(login_data: LoginRequest):
    """Login a user and return user data."""
    # Find user by email
    user = users_collection.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Return user data
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"]
    }

@app.get("/users", response_model=List[UserResponse])
async def get_all_users():
    """Get all users from the database."""
    users = []
    for user in users_collection.find():
        users.append({
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "created_at": user.get("created_at", datetime.utcnow())
        })
    return users

# Sample Questions Endpoints
@app.post("/sample-questions", response_model=SampleQuestionResponse)
async def add_sample_question(question_data: SampleQuestionCreate):
    """Add a new sample question (max 4)."""
    # Check if already at the limit
    questions_count = sample_questions_collection.count_documents({})
    if questions_count >= 4:
        raise HTTPException(
            status_code=400, 
            detail="Maximum of 4 sample questions are allowed. Delete an existing question first."
        )
    
    # Create new sample question
    question_dict = {
        "question": question_data.question,
        "created_at": datetime.utcnow()
    }
    
    result = sample_questions_collection.insert_one(question_dict)
    
    # Return the created question
    created_question = sample_questions_collection.find_one({"_id": result.inserted_id})
    return {
        "id": str(created_question["_id"]),
        "question": created_question["question"],
        "created_at": created_question["created_at"]
    }

@app.get("/sample-questions", response_model=List[SampleQuestionResponse])
async def get_sample_questions():
    """Get all sample questions."""
    questions = []
    for question in sample_questions_collection.find():
        questions.append({
            "id": str(question["_id"]),
            "question": question["question"],
            "created_at": question.get("created_at", datetime.utcnow())
        })
    return questions

@app.delete("/sample-questions/{question_id}")
async def delete_sample_question(question_id: str):
    """Delete a sample question by ID."""
    try:
        # Convert string ID to ObjectId
        object_id = ObjectId(question_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid question ID format")
    
    # Try to delete the question
    result = sample_questions_collection.delete_one({"_id": object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Sample question not found")
    
    return {"message": f"Sample question {question_id} deleted successfully"}

# User Feedback Endpoints (User-based feedback)
@app.post("/user-feedback", response_model=UserFeedbackResponse)
async def add_user_feedback(feedback_data: UserFeedbackCreate, user_id: str):
    """Add feedback from a user."""
    # Check if user exists
    try:
        user_obj_id = ObjectId(user_id)
        user = users_collection.find_one({"_id": user_obj_id})
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Create new feedback (tied to user only)
    feedback_dict = {
        "user_id": user_id,
        "rating": feedback_data.rating,
        "comment": feedback_data.comment,
        "created_at": datetime.utcnow()
    }
    
    # Insert new feedback
    result = user_feedback_collection.insert_one(feedback_dict)
    feedback_id = str(result.inserted_id)
    
    # Return the created feedback
    created_feedback = user_feedback_collection.find_one({"_id": ObjectId(feedback_id)})
    return {
        "id": feedback_id,
        "user_id": created_feedback["user_id"],
        "rating": created_feedback["rating"],
        "comment": created_feedback.get("comment"),
        "created_at": created_feedback["created_at"]
    }

@app.get("/user-feedback/{user_id}", response_model=List[UserFeedbackResponse])
async def get_user_feedback(user_id: str):
    """Get all feedback provided by a specific user."""
    feedback_list = []
    for feedback in user_feedback_collection.find({"user_id": user_id}):
        feedback_list.append({
            "id": str(feedback["_id"]),
            "user_id": feedback["user_id"],
            "rating": feedback["rating"],
            "comment": feedback.get("comment"),
            "created_at": feedback["created_at"]
        })
    return feedback_list

# Document Management Endpoints with View Support
@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    document_name: Optional[str] = Form(None)
):
    """
    Endpoint to upload a PDF file, extract text, and add it to the vector store.
    Uses the original filename as the document ID by default.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Create temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(temp_file_path)
        
        # Use original filename as document_id (without extension)
        if document_name:
            # If document_name is provided, use it
            document_id = document_name
        else:
            # Use the original filename (without .pdf extension)
            document_id = os.path.splitext(file.filename)[0]
            
        # Sanitize document ID (remove spaces, special characters)
        document_id = document_id.replace(" ", "_").lower()
        
        # Save the PDF file for viewing
        pdf_dir = "pdfs"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        
        pdf_path = os.path.join(pdf_dir, f"{document_id}.pdf")
        shutil.copy(temp_file_path, pdf_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Add document to vector store
        success = add_document_to_vector_store(pdf_text, document_id)
        
        if success:
            return PDFUploadResponse(
                message="PDF processed and added to knowledge base successfully",
                document_id=document_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to add document to knowledge base")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        file.file.close()

@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """
    Endpoint to list all documents in the knowledge base without ratings.
    """
    try:
        documents = []
        data_dir = "data"
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(".txt"):
                    # Remove the .txt extension
                    document_id = filename[:-4]
                    
                    documents.append({
                        "id": document_id,
                        "has_pdf": os.path.exists(os.path.join("pdfs", f"{document_id}.pdf"))
                    })
        
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/documents/{document_id}/content")
async def get_document_content(document_id: str):
    """
    Endpoint to get the extracted text content of a document.
    """
    try:
        file_path = os.path.join("data", f"{document_id}.txt")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Read the document content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {"document_id": document_id, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document content: {str(e)}")

@app.get("/documents/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """
    Endpoint to get the original PDF of a document.
    """
    try:
        pdf_path = os.path.join("pdfs", f"{document_id}.pdf")
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF for document {document_id} not found")
        
        return FileResponse(pdf_path, media_type="application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Endpoint to delete a document from the knowledge base and rebuild the index.
    """
    try:
        file_path = os.path.join("data", f"{document_id}.txt")
        pdf_path = os.path.join("pdfs", f"{document_id}.pdf")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Delete the document text file
        os.remove(file_path)
        
        # Delete the PDF file if it exists
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Rebuild the index
        await rebuild_index()
        
        return {"message": f"Document {document_id} deleted and index rebuilt successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """
    Endpoint to rebuild the vector store index.
    """
    global vector_store
    try:
        # Delete existing index if it exists
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        
        # Reinitialize vector store
        vector_store = initialize_vector_store()
        return {"message": "Vector store index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

# ChatBot Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that accepts a question and returns an answer with optional context.
    """
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        chat_chain = get_chat_chain(vector_store)
        response = chat_chain.invoke({"input": request.question})
        
        # Extract context content for response
        context_content = [doc.page_content for doc in response["context"]]
        
        return ChatResponse(
            answer=response["answer"],
            context=context_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)