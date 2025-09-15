# pip install google-generativeai python-dotenv pypdf pdf2image Pillow pytesseract sentence-transformers faiss-cpu PyMuPDF
import os
import io
import re
from dotenv import load_dotenv
from typing import List, Dict, Any

import google.generativeai as genai
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

import fitz
from PIL import Image
import io

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# SETUP
def configure_environment():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=api_key)

# PDF PARSING
def extract_elements_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    print(f"Processing PDF: {pdf_path}")
    elements = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        # Extract Text
        text = page.get_text()
        if text:
            text = re.sub(r'\s+', ' ', text).strip()
            elements.append({"type": "text", "content": text, "page": page_num + 1})

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert bytes to a PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            elements.append({"type": "image", "content": pil_image, "page": page_num + 1})

    print(f"Extracted {len(elements)} elements from PDF.")
    doc.close()
    return elements

# --- 3. MULTIMODAL PROCESSING (OCR & IMAGE CAPTIONING) ---
def process_elements(elements: List[Dict[str, Any]]) -> List[str]:
    """
    Processes extracted text and images.
    - Text is chunked.
    - Images are described using a multimodal model and OCR.
    """
    processed_texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    for element in elements:
        if element["type"] == "text":
            chunks = text_splitter.split_text(element["content"])
            for chunk in chunks:
                processed_texts.append(f"[Text from Page {element['page']}] {chunk}")
        
        elif element["type"] == "image":
            print(f"🖼️ Processing image from page {element['page']}...")
            img = element["content"]
            
            # Get image description
            try:
                response = vision_model.generate_content(["Describe this image, chart, or graph in detail.", img], stream=False)
                description = f"[Image Description from Page {element['page']}] {response.text}"
                processed_texts.append(description)
            except Exception as e:
                print(f"   - Vision model failed: {e}")

            # Get OCR text (for handwritten notes)
            try:
                # You might need to specify the path to the tesseract executable
                # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    processed_texts.append(f"[Handwritten/OCR Text from Page {element['page']}] {ocr_text.strip()}")
            except Exception as e:
                print(f"   - Tesseract OCR failed: {e}")

    print(f"🧠 Processed into {len(processed_texts)} text chunks.")
    return processed_texts

# --- 4. EMBEDDING AND VECTOR DATABASE INDEXING ---
class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, texts: List[str]):
        """Creates embeddings and builds a FAISS index."""
        print("🚀 Building vector index...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents = texts
        print(f"✅ Index built successfully with {len(self.documents)} documents.")

    def search(self, query: str, k: int = 5) -> List[str]:
        """Searches the index for the most relevant documents."""
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.documents[i] for i in indices[0]]

# --- 5. RAG CORE (RETRIEVAL AND GENERATION) ---
class MRAGSystem:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.generator_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _get_context(self, query: str) -> str:
        """Retrieves relevant context from the vector store."""
        retrieved_docs = self.vector_store.search(query, k=5)
        return "\n---\n".join(retrieved_docs)

    def answer_query(self, query: str) -> str:
        """Answers a query based on the content of the PDF."""
        print(f"\n❓ Answering Query: {query}")
        context = self._get_context(query)
        
        prompt = f"""
        **Context from the document:**
        {context}

        **Question:**
        {query}

        Based *only* on the provided context, please provide a clear and concise answer. If the context does not contain the answer, state that.
        """
        
        response = self.generator_model.generate_content(prompt)
        return response.text

    def generate_presentation(self, topic: str, num_slides: int = 5) -> str:
        """Generates a presentation outline on a given topic from the PDF."""
        print(f"\n🎬 Generating Presentation on: {topic}")
        context = self._get_context(topic)
        
        prompt = f"""
        **Context from the document related to '{topic}':**
        {context}

        **Task:**
        Create a {num_slides}-slide presentation outline based *only* on the provided context.
        For each slide, provide a title and 3-4 bullet points.
        
        **Format:**
        Slide 1: [Title]
        - Bullet point 1
        - Bullet point 2
        
        Slide 2: [Title]
        - ...
        """
        
        response = self.generator_model.generate_content(prompt)
        return response.text

    def generate_study_plan(self, subject: str, days: int = 3) -> str:
        """Generates a study plan for a subject based on the PDF."""
        print(f"\n📅 Generating Study Plan for: {subject}")
        context = self._get_context(subject)
        
        prompt = f"""
        **Context from the document about '{subject}':**
        {context}
        
        **Task:**
        Create a {days}-day study plan based *only* on the provided context.
        For each day, identify key topics to cover and suggest one review activity.

        **Format:**
        Day 1:
        - Topic: [Key topic 1 from context]
        - Topic: [Key topic 2 from context]
        - Activity: [e.g., 'Summarize the main points of X']

        Day 2:
        - ...
        """
        
        response = self.generator_model.generate_content(prompt)
        return response.text

# --- 6. MAIN EXECUTION ---
if __name__ == '__main__':
    # Ensure you have a PDF file named 'sample.pdf' in the same directory
    pdf_file = "Springer_EmbeddingsRP.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: The file '{pdf_file}' was not found.")
        print("Please add a PDF file to your project directory and name it 'sample.pdf'.")
    else:
        # --- Full Pipeline ---
        # 1. Setup
        configure_environment()
        
        # 2. & 3. Extract and Process
        elements = extract_elements_from_pdf(pdf_file)
        processed_content = process_elements(elements)
        
        # 4. Index
        vector_db = VectorStore()
        vector_db.build_index(processed_content)
        
        # 5. Initialize RAG System
        mrag = MRAGSystem(vector_db)
        
        # --- 6. Interact with the system ---
        print("\n" + "="*50)
        print("      Multimodal RAG System is Ready!")
        print("="*50)
        
        # Example 1: Ask a direct question
        question_1 = "What is the main conclusion of the document?"
        answer_1 = mrag.answer_query(question_1)
        print(f"\n✅ Answer to '{question_1}':\n{answer_1}")

        # Example 2: Ask a question about an image/graph
        question_2 = "Summarize the trend shown in the bar chart on page 2."
        answer_2 = mrag.answer_query(question_2)
        print(f"\n✅ Answer to '{question_2}':\n{answer_2}")
        
        # Example 3: Generate a presentation
        presentation_topic = "The key findings about market growth"
        presentation = mrag.generate_presentation(presentation_topic)
        print(f"\n✅ Presentation on '{presentation_topic}':\n{presentation}")

        # Example 4: Generate a study plan
        study_subject = "The entire document"
        study_plan = mrag.generate_study_plan(study_subject)
        print(f"\n✅ Study Plan for '{study_subject}':\n{study_plan}")