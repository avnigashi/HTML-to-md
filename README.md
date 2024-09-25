# HTML to Markdown Converter

This application provides a graphical user interface for converting HTML content to Markdown using the Jina Reader-LM model. It's built with Python, PyQt5 for the GUI, and the Hugging Face Transformers library for the conversion logic.

## Repository

https://github.com/avnigashi/HTML-to-md

## Features

- Simple and intuitive GUI
- Utilizes the Jina Reader-LM 1.5b model for accurate HTML to Markdown conversion
- Real-time conversion with a single click

## Requirements

- Python 3.7+
- PyQt5
- transformers
- torch

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/avnigashi/HTML-to-md.git
   cd HTML-to-md
   ```

2. It's recommended to create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. The application window will open.

3. Enter or paste your HTML content into the top text area.

4. Click the "Convert to Markdown" button.

5. The converted Markdown will appear in the bottom text area.
