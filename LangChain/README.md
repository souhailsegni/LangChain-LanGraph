# LangChain-LanGraph

A small Streamlit app that generates polished articles from a user-provided topic using Google's Gemini model.

## Features
- Simple article generator interface
- Uses Gemini via LangChain
- Reads the API key from a `.env` file
- Clean markdown output for the generated article

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project files
- `app.py` — Streamlit app logic
- `.env` — Local environment file holding your API key
- `requirements.txt` — Python dependencies

## Notes
- Do not commit your real `.env` file to Git.
- A sample template is available in `.env.example` if you want to copy it.