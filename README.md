# Shahim Quraishy – Résumé

A personal résumé website with an AI-powered chat assistant that answers recruiter questions based on the CV.

## Features

- **Interactive résumé** – Modern, responsive HTML page (`index.html`)
- **AI chat assistant** – FastAPI backend powered by Google Gemini that answers questions about Shahim's experience and skills
- **PDF download** – Downloadable PDF version of the CV (`lebenslauf.pdf`)

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI, LangChain
- **AI:** Google Gemini (`gemini-2.5-flash`)

## Getting Started

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
pip install -r requirements.txt
```

### Running the server

```bash
export GOOGLE_API_KEY=your_api_key_here
python start.py
```

The API will be available at `http://localhost:10000`.

### Endpoints

| Method | Path    | Description                              |
|--------|---------|------------------------------------------|
| GET    | `/`     | Health check                             |
| POST   | `/chat` | Ask the AI assistant a question about Shahim's CV |

#### Example request

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -d '{"frage": "What are Shahim'\''s key skills?"}'
```

## Project Structure

```
CV/
├── index.html        # Résumé website (frontend)
├── main.py           # FastAPI application
├── start.py          # Server entry point
├── requirements.txt  # Python dependencies
├── runtime.txt       # Python version
├── lebenslauf.pdf    # PDF version of the CV
└── profilbild.jpg    # Profile photo
```
