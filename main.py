import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
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

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize FastAPI app
app = FastAPI(title="CUTM Chatbot API")

# Initialize global vector store
vector_store = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None

class PDFUploadResponse(BaseModel):
    message: str
    document_id: str

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

@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    global vector_store
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    vector_store = initialize_vector_store()

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

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...), document_name: Optional[str] = Form(None)):
    """
    Endpoint to upload a PDF file, extract text, and add it to the vector store.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Generate document ID (use provided name or generate UUID)
        document_id = document_name if document_name else str(uuid.uuid4())
        document_id = document_id.replace(" ", "_").lower()  # Sanitize document ID
        
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

@app.get("/documents", response_model=List[str])
async def list_documents():
    """
    Endpoint to list all documents in the knowledge base.
    """
    try:
        documents = []
        data_dir = "data"
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(".txt"):
                    # Remove the .txt extension
                    document_id = filename[:-4]
                    documents.append(document_id)
        
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Endpoint to delete a document from the knowledge base and rebuild the index.
    """
    try:
        file_path = os.path.join("data", f"{document_id}.txt")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Delete the document
        os.remove(file_path)
        
        # Rebuild the index
        await rebuild_index()
        
        return {"message": f"Document {document_id} deleted and index rebuilt successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)