import os
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import cohere
import re
import gradio as gr
import io
import random
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from io import BytesIO

# Initialize PaddleOCR and Cohere Client
paddle_ocr = PaddleOCR()

# Fetch Cohere API key
api_key = "YOUR_KEY"  # Replace with your actual API key
if not api_key:
    raise ValueError("Cohere API key not provided.")
co = cohere.Client(api_key)

# Configure Cloudinary account
cloudinary.config(
    cloud_name="NAME",  # Replace with your Cloudinary cloud name
    api_key="API_KEY",        # Replace with your Cloudinary API key
    api_secret="API_SECRET"   # Replace with your Cloudinary API secret
)

def download_image_from_url(url):
    """Download image from URL and return as PIL Image object"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def fetch_from_cloudinary(folder_prefix):
    """Fetch image URLs from Cloudinary based on folder name"""
    try:
        response = cloudinary.api.resources(
            type="upload",
            prefix=folder_prefix,
            resource_type="image",
            max_results=100
        )
        return [resource["secure_url"] for resource in response.get("resources", [])]
    except Exception as e:
        return f"Failed to fetch files from Cloudinary: {e}"

def process_documents(document_type, image_sources, is_url=False):
    """Process documents from either uploaded files or URLs"""
    extracted_data = []
    for source in image_sources:
        try:
            # Handle both uploaded files and URLs
            if is_url:
                image = download_image_from_url(source)
                if image is None:
                    continue
            else:
                image = Image.open(source)

            # Convert PIL Image to CV2 format
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Perform OCR
            paddle_results = paddle_ocr.ocr(image_cv2)

            # Extract text from OCR results
            if paddle_results and paddle_results[0]:
                texts = [line[1][0] for line in paddle_results[0]]
                extracted_text = " ".join(texts)

                # Generate prompt based on document type
                prompt = f"Extract the following information from the text: {extracted_text}\n\n"
                if document_type == "Salary Slip":
                    prompt += "Find and return the values for: Net Salary, Gross Salary, Basic Salary."
                elif document_type == "Profit and Loss Statement":
                    prompt += "Find and return the values for: Total Revenue, Net Income."
                elif document_type == "Check":
                    prompt += "Find and return the values for: Account Number, Amount (in Rupees), Bank Name."

                # Get response from Cohere
                response = co.generate(model='command', prompt=prompt, max_tokens=200)
                result_text = response.generations[0].text.strip()

                # Parse the response into a dictionary
                temp_data = {}
                for line in result_text.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        temp_data[key.strip()] = value.strip()
                extracted_data.append(temp_data)
            else:
                extracted_data.append({"Error": "No text detected in image"})
        except Exception as e:
            extracted_data.append({"Error": f"Failed to process image: {str(e)}"})
    return extracted_data

def visualize_results(document_type, image_sources, plot_type, is_url=False):
    """Visualize results from processed documents"""
    if not image_sources:
        return "No files provided. Please upload images or fetch from Cloudinary.", None

    extracted_data = process_documents(document_type, image_sources, is_url)
    
    if not extracted_data:
        return "No data extracted. Please check your input files.", None

    tables_html = ""
    visualization_images = []

    for idx, data in enumerate(extracted_data):
        if data:
            df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
            tables_html += f"<h4>Extracted Information for Image {idx + 1}</h4>"
            tables_html += df.to_html(index=False, escape=False, border=0)

            # Prepare data for visualization for each image separately
            flattened_data = {}
            for key, value in data.items():
                if key != "Error":  # Skip error messages
                    try:
                        numeric_value = float(re.sub(r"[^\d.]+", "", value))
                        flattened_data[key] = numeric_value
                    except ValueError:
                        pass
            
            if flattened_data:  # Create visualization for this specific image
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = list(flattened_data.keys())
                values = list(flattened_data.values())
                
                if plot_type == "Bar Plot":
                    ax.bar(labels, values, color="skyblue", edgecolor="black")
                    ax.set_title(f"{document_type} - Image {idx + 1} - Bar Plot")
                    ax.set_ylabel("Amount")
                    ax.set_xlabel("Fields")
                    plt.xticks(rotation=45, ha="right")
                
                elif plot_type == "Pie Chart":
                    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
                           colors=plt.cm.Paired.colors)
                    ax.set_title(f"{document_type} - Image {idx + 1} - Pie Chart")

                # Save plot to buffer 
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)
                visualization_images.append(image)
                
            else:  # If no valid data for visualization, append a blank placeholder image 
                blank_image = Image.new("RGB", (300, 200), color=(255, 255, 255))
                visualization_images.append(blank_image)
        else:
            tables_html += f"<h4>Image {idx + 1}</h4>"
            tables_html += "<p>No data extracted from this image.</p>"
            
            blank_image = Image.new("RGB", (300, 200), color=(255, 255, 255))
            visualization_images.append(blank_image)

    return tables_html, visualization_images

def process_cloudinary_images(document_type, num_images, plot_type):
    """Process images fetched from Cloudinary based on document type"""
    folder_mapping = {
        "Profit and Loss Statement": "dataset1/profit and loss",
        "Salary Slip": "dataset1/salary slip"
    }

    folder_name = folder_mapping.get(document_type)
    
    if folder_name is None:
        return "Invalid document type selected.", None, None

    urls = fetch_from_cloudinary(folder_name)  # Pass the correct folder name here
    
    if isinstance(urls, str):  # Error message 
        return urls, None, None
    
    selected_urls = random.sample(urls, min(len(urls), int(num_images)))
    
    gallery_images = [download_image_from_url(url) for url in selected_urls]
    gallery_images = [np.array(img) for img in gallery_images if img is not None]

    tables_html, visualization_images = visualize_results(document_type, selected_urls, plot_type, is_url=True)

    return tables_html, visualization_images, gallery_images

def launch_ui():
    """Launch the Gradio UI"""
    with gr.Blocks() as demo:
        gr.Markdown("# Document OCR, Analysis, and Visualization")
        
        with gr.Row():
            document_type = gr.Radio(
                ["Salary Slip", "Profit and Loss Statement", "Check"],
                label="Select Document Type",
            )
            plot_type = gr.Radio(["Bar Plot", "Pie Chart"], label="Select Plot Type")

        with gr.Tabs():
            with gr.TabItem("Upload Files"):
                uploaded_files = gr.File(file_types=["image"], file_count="multiple", label="Upload Document Images")
                upload_submit = gr.Button("Process Uploaded Files")

            with gr.TabItem("Cloudinary Images"):
                num_images = gr.Number(label="Number of Random Images from Cloudinary", value=1)
                fetch_button = gr.Button("Fetch and Process Cloudinary Images")
                
                cloudinary_gallery = gr.Gallery(label="Fetched Images from Cloudinary")
                
                result = gr.HTML(label="Extracted Information")
                
                visualization_output = gr.Gallery(label="Visualizations")  # Updated to display multiple visualizations

        def process_uploaded_images(document_type, uploaded_files, plot_type):
            image_sources = [file.name for file in uploaded_files]  # Convert to list of filenames 
            tables_html, visualization_images = visualize_results(document_type, image_sources, plot_type)
            return tables_html, visualization_images
        
        upload_submit.click(
            process_uploaded_images,
            inputs=[document_type, uploaded_files, plot_type],
            outputs=[result, visualization_output]
        )

        fetch_button.click(
            process_cloudinary_images,
            inputs=[document_type, num_images, plot_type],
            outputs=[result, visualization_output, cloudinary_gallery]
        )

    demo.launch()  # Run the pipeline

if __name__ == "__main__":
    launch_ui()