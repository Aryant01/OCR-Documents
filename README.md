# Document OCR, Analysis, and Visualization

## Overview
This project provides an Optical Character Recognition (OCR) and data visualization tool for processing various financial documents, including salary slips, profit and loss statements, and checks. It uses PaddleOCR for text extraction, Cohere for text analysis, Cloudinary for cloud-based image storage, and Gradio for an interactive UI.

## Features
- Extract text from images using PaddleOCR.
- Analyze and extract key financial data using Cohere's AI.
- Fetch and process images from Cloudinary.
- Generate visualizations (Bar Plot and Pie Chart) based on extracted data.
- Interactive web UI using Gradio.

## Dependencies
Ensure you have the following Python packages installed:

```sh
pip install paddleocr numpy opencv-python pandas matplotlib cohere gradio cloudinary requests Pillow
```

## Installation
1. Clone the repository:
   ```sh
   git clone git@github.com:Aryant01/OCR-Documents.git
   cd your-repository
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up API keys:
   - Replace `api_key` in the script with your actual Cohere API key.
   - Replace `cloud_name`, `api_key`, and `api_secret` with your Cloudinary credentials.

## Usage
To start the application, run:
```sh
python script.py
```

### UI Features
1. **Upload Files**:
   - Users can upload document images for processing.
   - Select the document type and visualization type (Bar Plot or Pie Chart).
2. **Fetch from Cloudinary**:
   - Retrieve images from Cloudinary for automatic processing.
   - Specify the number of images to fetch.
3. **Results**:
   - Extracted text is displayed in a table format.
   - Visual representations are generated for numerical data.

## Cloudinary Integration
This project can fetch images from Cloudinary. Ensure that your Cloudinary account is set up and the correct folder names are used.

## File Structure
```
project-directory/
├── script.py       # Main script containing the OCR and visualization logic
├── README.md       # Documentation (this file)
├── requirements.txt # Dependencies list
└── images/         # (Optional) Local storage for test images
