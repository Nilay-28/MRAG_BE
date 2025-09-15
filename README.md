# MRAG_BE

# 🧠 Multimodal RAG Learning Assistant

A simple yet powerful learning assistant built with **Streamlit** and **Google Gemini**. Chat with your study materials—including PDFs with text, images, and even handwritten notes—and get clear, sourced answers.

---

## ✨ Key Features

* **Multimodal Chat**: Interact with both PDF documents and images (`.png`, `.jpg`, `.jpeg`).
* **Advanced OCR**: Gemini's powerful vision model extracts text from images, including diagrams and handwritten notes.
* **Sourced Answers**: Never lose track of where information comes from. Every answer is backed by the specific text snippets and images from your documents.
* **Vector-Powered Search**: Utilizes Google's embedding models and a FAISS vector store for fast and contextually relevant information retrieval.
* **Intuitive Interface**: A clean and easy-to-use web interface powered by Streamlit.

---

## ⚙️ How It Works

This application uses a Retrieval-Augmented Generation (RAG) pipeline to understand and answer questions about your documents.

1.  **📤 Ingestion**: You upload your PDF files and images through the web interface.
2.  **🔍 Extraction & OCR**: The app processes PDFs to extract both text and embedded images. For every image, Google Gemini performs OCR to transcribe any visible text.
3.  **🧠 Vectorization**: All extracted text is chunked and converted into numerical vectors (embeddings) using Google's `text-embedding-004` model. These are stored in an efficient FAISS vector store.
4.  **💬 Generation**: When you ask a question, the app:
    * Searches the vector store for the most relevant text chunks.
    * Identifies relevant source images.
    * Passes this rich context (text + images) to the Gemini model to generate a comprehensive, context-aware answer.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* A Google API Key with the Gemini API enabled.

### Installation

Clone the repository and install the required Python packages.

```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository-directory>

# Install dependencies
pip install -r requirements.txt

# .env
GOOGLE_API_KEY="YOUR_API_KEY_HERE"

# Libraries
pip install google-generativeai python-dotenv pypdf pdf2image Pillow pytesseract sentence-transformers faiss-cpu PyMuPDF
```
### Execution

Execute The File.
```bash
python app.py
```
