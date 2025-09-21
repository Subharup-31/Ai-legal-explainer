# AI Legal Explainer

This project is a sophisticated web application that acts as an AI-powered legal assistant. It helps users understand complex legal documents by providing plain-language summaries, detailed clause explanations, and risk assessments. The application leverages a Retrieval-Augmented Generation (RAG) architecture, integrating powerful AI and cloud services to deliver a seamless and intelligent user experience.

## Features

- **High-Accuracy Text Extraction**: Utilizes **Google Cloud Document AI** to accurately extract text from uploaded PDF documents, including scanned files and complex layouts.
- **Advanced Semantic Search**: Employs vector embeddings and a **Pinecone** vector database to find the most semantically relevant clauses in a document based on the user's query.
- **Structured AI Explanations**: Uses **Google's Gemini AI** to generate structured, easy-to-understand explanations of legal text, including:
  - A direct quote of the relevant clause.
  - A plain-language paraphrase.
  - A color-coded risk assessment (Green, Amber, Red).
  - Practical next steps and suggested follow-up questions.
- **Multi-Language Support**: Automatically detects the language of a user's question and responds in the same language. Supported languages include English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, and Hindi.
- **Interactive Web Interface**: Built with **Streamlit** to provide a clean, modern, and user-friendly interface for document management and chat.
- **Secure Authentication**: Uses a Google Cloud service account for secure, server-side authentication.

## Technology Stack

- **Backend**: Python
- **Frontend**: Streamlit
- **Generative AI**: Google Gemini
- **Vector Database**: Pinecone
- **Text Extraction (OCR)**: Google Cloud Document AI
- **Embeddings**: Sentence-Transformers

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8+
- A Google Cloud Platform (GCP) project with the Document AI API enabled.
- A Pinecone account.

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of the project and add the following variables:

```dotenv
# Google Gemini API Key
GOOGLE_API_KEY="your-gemini-api-key"

# Pinecone API Key
PINECONE_API_KEY="your-pinecone-api-key"

# --- Google Cloud Document AI Configuration ---

# 1. Your Google Cloud Project ID
DOCAI_PROJECT_ID="your-gcp-project-id"

# 2. The location of your Document AI processor (e.g., 'us' or 'eu')
DOCAI_LOCATION="your-docai-location"

# 3. The ID of your Document AI processor
DOCAI_PROCESSOR_ID="your-docai-processor-id"

# 4. Path to your Google Cloud service account key file
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### 5. Set Up Google Cloud Authentication

- **Create a Service Account**: In your GCP project, create a service account with the "Document AI API User" role.
- **Generate a Key**: Create a JSON key for this service account and download it.
- **Update `.env`**: Place the downloaded JSON key file in your project directory and set the `GOOGLE_APPLICATION_CREDENTIALS` variable in your `.env` file to its path.

### 6. Running the Application

Once all the dependencies are installed and the environment variables are configured, run the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.
