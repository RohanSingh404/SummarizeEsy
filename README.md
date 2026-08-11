# 📝 Text Summarization using T5, Hugging Face & FastAPI

## 📌 Overview

This project is a **Text Summarization application** built using **FastAPI** and the **Hugging Face Transformers** library.

The application uses a **T5 (Text-to-Text Transfer Transformer)** model to generate concise summaries from input text or dialogue. The trained model is loaded locally and exposed through a FastAPI backend, making the summarization model accessible through an API as well as a simple web interface.

The project demonstrates how to take a Transformer-based NLP model and turn it into a usable **ML-powered web application/API**.

---

## 🚀 Features

* 🤗 **Hugging Face Transformers** for NLP
* 🧠 **T5 model** for text summarization
* ⚡ **FastAPI** backend
* 🌐 Simple web interface using Jinja2 templates
* 🔌 REST API endpoint for text summarization
* 🧹 Input text preprocessing and cleaning
* 🖥️ Automatic device selection:

  * Apple Silicon (`MPS`)
  * NVIDIA GPU (`CUDA`)
  * CPU
* ✂️ Input truncation and tokenization using the T5 tokenizer
* 🔥 Beam Search generation for better summaries

---

## 🏗️ Project Architecture

```text
                 ┌─────────────────────┐
                 │     User Input      │
                 │   Text / Dialogue   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      FastAPI        │
                 │      Backend        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Text Cleaning     │
                 │  - Remove HTML      │
                 │  - Normalize spaces │
                 │  - Lowercase text   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  T5 Tokenizer       │
                 │  Max Length: 512    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     T5 Model        │
                 │  Text Summarization │
                 │   Beam Search = 4   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Summary         │
                 └─────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

* **Python**
* **FastAPI**
* **Pydantic**
* **Jinja2**

### Machine Learning / NLP

* **Hugging Face Transformers**
* **T5ForConditionalGeneration**
* **T5Tokenizer**
* **PyTorch**

### Other Libraries

* **Regex (`re`)**
* **HTMLResponse**
* **StaticFiles**

---

## 📂 Project Structure

```text
text-summarizer/
│
├── app.py
├── index.html
│
├── saved_summary_model/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── tokenizer files
│   └── model weights
│
└── README.md
```

> The `saved_summary_model/` directory contains the locally saved T5 summarization model and tokenizer used by the application.

---

## ⚙️ How It Works

### 1. Load the Model

The application loads the trained/saved T5 model and tokenizer from the local `saved_summary_model` directory.

```python
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
```

---

### 2. Select the Computing Device

The application checks whether hardware acceleration is available.

```text
Apple Silicon → MPS
       ↓
NVIDIA GPU → CUDA
       ↓
CPU
```

This allows the model to run on available hardware.

---

### 3. Clean the Input

Before summarization, the input text is cleaned using regular expressions.

The preprocessing includes:

* Removing line breaks
* Normalizing multiple spaces
* Removing HTML tags
* Removing leading/trailing whitespace
* Converting text to lowercase

Example:

```text
"Hello!\n\nThis is    some text."
                ↓
"hello! this is some text."
```

---

### 4. Tokenization

The cleaned dialogue is converted into tokens using the T5 tokenizer.

The application uses:

```python
max_length=512
truncation=True
padding="max_length"
```

This ensures that the input passed to the model has a controlled maximum length.

---

### 5. Generate Summary

The T5 model generates the summary using beam search.

```python
targets = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    num_beams=4,
    early_stopping=True
)
```

The generated token IDs are then decoded back into human-readable text.

---

## 🔌 API Endpoint

### `POST /summarize/`

This endpoint accepts text/dialogue and returns its generated summary.

### Request

```json
{
    "dialogue": "Your long text or dialogue goes here..."
}
```

### Response

```json
{
    "summary": "Generated summary of the input text."
}
```

---

## 🧪 Running the Project Locally

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd text-summarizer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install fastapi uvicorn transformers torch jinja2 pydantic
```

---

### 4. Make Sure the Model Exists

The application expects the saved model at:

```text
./saved_summary_model
```

Make sure the model and tokenizer files are present in this directory.

---

### 5. Start the FastAPI Server

```bash
uvicorn app:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

## 📖 API Documentation

FastAPI automatically provides interactive API documentation.

After starting the server, visit:

```text
http://127.0.0.1:8000/docs
```

You can use the Swagger UI to test the `/summarize/` endpoint directly from your browser.

Alternative documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## 💡 Example

### Input

```text
John and Sarah discussed the upcoming project.
John explained that the project needs to be completed
within two weeks. Sarah agreed and said that she would
handle the documentation while John worked on the backend.
```

### Output

```text
John and Sarah discussed the project timeline and divided
the work, with Sarah handling documentation and John
working on the backend.
```

---

## 🧠 Model

The application uses:

**T5 — Text-to-Text Transfer Transformer**

T5 treats NLP problems as text-to-text tasks. In this project, the model receives the input dialogue/text and generates a shorter summarized version.

The model is loaded using:

```python
T5ForConditionalGeneration
```

and tokenized using:

```python
T5Tokenizer
```

from the Hugging Face Transformers ecosystem.

---

## 🔍 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Natural Language Processing (NLP)
* Text Summarization
* Transformer Models
* T5 Architecture
* Hugging Face Transformers
* Tokenization
* Text preprocessing
* Sequence generation
* Beam Search
* PyTorch inference
* REST APIs
* FastAPI
* API request validation with Pydantic
* Model deployment architecture

---

## 📈 Possible Improvements

Some possible improvements for the project include:

* Add support for larger input documents
* Implement chunk-based summarization for long documents
* Add summary length controls
* Add multiple summarization models
* Add authentication for the API
* Add request rate limiting
* Add Docker support
* Deploy the API using a cloud platform
* Deploy the model using Hugging Face Spaces or an inference service
* Add logging and monitoring
* Add unit and integration tests
* Create a more advanced frontend
* Add PDF/TXT document upload and summarization

---

## 🎯 Future Scope

The project can be extended into a complete **AI-powered document summarization platform**.

Potential features include:

```text
PDF Upload
     ↓
Text Extraction
     ↓
Text Preprocessing
     ↓
Chunking
     ↓
T5 / Transformer Model
     ↓
Summary Generation
     ↓
Download / Share Summary
```

This would make the project more suitable for real-world document processing applications.

---

## 👨‍💻 Author

**Rohan Singh**

GitHub: https://github.com/RohanSingh404

LinkedIn: https://linkedin.com/in/rohansingh404

---

⭐ If you found this project useful, consider giving it a star!
