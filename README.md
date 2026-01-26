# Raheem AI - Local Development

This project provides a web-based chat interface to ask questions about a collection of PDF documents, with a focus on Irish Building Regulations.

## Local Development Setup

### 1. Prerequisites

- Python 3.10+
- An environment that can install Python packages (e.g., using `venv`).

### 2. Configure Environment Variables

The application uses a `.env` file for configuration and secrets. Start by copying the example file:

```bash
cp .env.example .env
```

Now, open the `.env` file in your editor and fill in the required values, especially your `GCP_PROJECT_ID` and `GCP_LOCATION`.

To authenticate with Google Cloud for local development, the easiest method is to use the `gcloud` CLI:
```bash
gcloud auth application-default login
```

### 3. Install Dependencies & Run

First, create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the required packages:
```bash
pip install -r requirements.txt
```

Once dependencies are installed, you can start the web server:
```bash
uvicorn main:app --reload
```

You should now be able to access the application at [http://127.0.0.1:8000](http://127.0.0.1:8000). The server hosts the frontend automatically.
