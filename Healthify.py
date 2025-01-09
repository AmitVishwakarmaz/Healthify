import os
import cv2
from pyzbar.pyzbar import decode
import pytesseract
from PIL import Image
import pandas as pd
import re
import numpy as np
import requests
import gradio as gr
import json
import threading
import time

# Set the PATH to include the directory containing libzbar-64.dll
os.environ['PATH'] = r'C:\zbar\bin;' + os.environ['PATH']

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

myconfig = r"--psm 11 --oem 3"

# OCR class from the first code
class OCR:
    def __init__(self):
        self.path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = self.path

    def enhance_image(self, image_path):
        image = Image.open(image_path)
        image_np = np.array(image)

        def adjust_brightness_contrast(image, brightness=0, contrast=30):
            img = np.int16(image)
            img = img * (contrast / 127 + 1) - contrast + brightness
            img = np.clip(img, 0, 255)
            return np.uint8(img)

        enhanced_image = adjust_brightness_contrast(image_np)

        def sharpen(image):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced_image, -1, kernel)
            return sharpened

        sharpened_image = sharpen(enhanced_image)
        denoised_image = cv2.fastNlMeansDenoisingColored(sharpened_image, None, 10, 10, 7, 21)
        enhanced_pil_image = Image.fromarray(denoised_image)
        return enhanced_pil_image

    def extract(self, image):
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(e)
            return "error"

    def check_image_quality(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var

ocr = OCR()

# Function to process the ingredients from text using OCR
def process_ingredients_text(text):
    text = text.replace("Ingredients:", "").replace("\n", " ")
    text = text.replace("INGREDIENTS:", "").replace("\n", " ")

    pattern = r',\s*(?![^()]*\))'
    ingredients = re.split(pattern, text)

    ingredient_names = []
    details_list = []
    percentages = []

    percentage_pattern = r'\(\d+(\.\d+)?%\)'

    for ingredient in ingredients:
        parenthesis_pattern = r'([^(]+)\s*\(([^)]+)\)'
        match = re.match(parenthesis_pattern, ingredient.strip())
        if match:
            main_part = match.group(1).strip()
            details = match.group(2).strip()
            
            percentage_match = re.search(percentage_pattern, f'({details})')
            if (percentage_match):
                percentage = percentage_match.group(0)
                details = re.sub(percentage_pattern, '', details).strip()
                percentages.append(percentage.strip('()'))
                details_list.append('')
            else:
                percentages.append('')
                details_list.append(details)
            
            ingredient_names.append(main_part)
        else:
            ingredient_names.append(ingredient.strip())
            details_list.append('')
            percentages.append('')

    df = pd.DataFrame({
        'Ingredient': ingredient_names,
        'Details': details_list,
        'Percentage': percentages
    })

    return df

def get_ingredients_from_open_food_facts(barcode_data):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_data}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        product_data = response.json()
        if product_data.get("status") == 1:
            product_name = product_data['product'].get('product_name', 'Unknown Product')
            ingredients_text = product_name + '\n' + product_data['product'].get('ingredients_text', 'No ingredients information found.')
            return ingredients_text, product_name
        else:
            return "Product not found in Open Food Facts database.", ""
    else:
        return "Error fetching data from Open Food Facts API.", ""

def generate_text(input_text):
    api_key = "" # Update with your actual API key
    endpoint = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'

    body = json.dumps({
        "contents": [{
            "parts": [{
                "text": input_text
            }]
        }]
    })

    try:
        response = requests.post(endpoint, headers={'Content-Type': 'application/json'}, data=body)
        response_data = response.json()
        text_part = response_data['candidates'][0]['content']['parts'][0]['text']
        return text_part
    except Exception as e:
        print("Error:", e)
        return None

def process_input(product_name):
    if product_name.lower() == 'exit':
        return "Exiting..."
    else:
        prompt = "List the ingredients, ingredient components, health hazards, allergen information, and whether it is veg or non-veg of " + product_name + " and do not provide any extra information"
        response = generate_text(prompt)
        print(response)
        if response:
            return response
        else:
            return "An error occurred while processing the input."

def barcode123(barcode):
    api_url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    
    try:
        response = requests.get(api_url)
        data = response.json()
        
        if "product" in data:
            product_info = data["product"]
            product_name = product_info.get("product_name", "Unknown Product")
            ingredients = product_info.get("ingredients_text", "Ingredients not available")
            
            if ingredients != "Ingredients not available":
                df = process_ingredients_text(ingredients)
                return f"Product Name: {product_name}\n\n{df.to_string(index=False)}"
            else:
                return f"Product Name: {product_name}\n\nIngredients not available"
        else:
            return "Product not found for the given barcode."
    except requests.RequestException as e:
        return f"Error fetching data: {e}"

def scan_barcode_info():
    cap = cv2.VideoCapture(0)
    product_result = ""

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detect_barcode = decode(frame)

        for barcode in detect_barcode:
            if barcode.data != "":
                barcode_data = barcode.data.decode('utf-8')
                result_text, product_name = get_ingredients_from_open_food_facts(barcode_data)
                cap.release()
                cv2.destroyAllWindows()
                product_result = process_input(product_name)
                return product_result
        
        cv2.imshow('Scanner', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return product_result

def scan_barcode_ingred():
    cap = cv2.VideoCapture(0)
    result_text = ""

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detect_barcode = decode(frame)

        for barcode in detect_barcode:
            if barcode.data != "":
                barcode_data = barcode.data.decode('utf-8')
                result_text, product_name = get_ingredients_from_open_food_facts(barcode_data)
                detailed_info1 = process_input(result_text)
                cap.release()
                cv2.destroyAllWindows()
                return f"Result: {result_text}\n\nDetailed Information: {detailed_info1}"
        
        cv2.imshow('Scanner', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return f"Result: {result_text}\n\nDetailed Information: {detailed_info1}"

def process_ocr_image(image_path):
    quality = ocr.check_image_quality(image_path)
    if quality < 100:
        return "The image quality is too low. Please upload a higher quality image."
    else:
        enhanced_image = ocr.enhance_image(image_path)
        text = ocr.extract(enhanced_image)
        return text

def process_and_summarize_ocr_image(image_path):
    ocr_result = process_ocr_image(image_path)
    detailed_info = process_input(ocr_result)
    return f"OCR Result: {ocr_result}\n\nDetailed Information: {detailed_info}"




with gr.Blocks() as demo:
    gr.Markdown("# Healthify")

    # First section: Food Name input and button below it
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines=5, placeholder="Enter the name of the food item...", label="Food Name")
            User_Input = gr.Textbox(label="Based on User Input")
            User_input_Button = gr.Button("Generate Info Based on User Input")
        
    gr.Markdown("<br>") 
    gr.Markdown("<br>") 
    

    # Second section: Barcode info scan and textboxes aligned with buttons below them
    with gr.Row():
        with gr.Column():
            Detailed_Info_Barcode = gr.Textbox(label="Basic Overall Information of Item (Barcode)")
            Detailed_Info_Barcode_Buton = gr.Button("Scan Barcode for Basic Overall Information")

    gr.Markdown("<br>") 
    gr.Markdown("<br>") 

    with gr.Row():
        with gr.Column():
            Detailed_InfoOfIngredient_Barcode = gr.Textbox(label="Detailed Ingredients Information of Item (Barcode)")
            Detailed_InfoOfIngredient_Barcode_Button = gr.Button("Scan Barcode for Detailed Ingredients Information")

    gr.Markdown("<br>") 
    gr.Markdown("<br>") 

    # Third section: OCR and file upload
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload an image containing text")
            ocr_output_box = gr.Textbox(label="OCR Result with Detailed Information")
            upload_button = gr.Button("Upload Image for OCR")

    gr.Markdown("<br>") 
    gr.Markdown("<br>")             

    # Clear button at the bottom
    clear_button = gr.Button("Clear All Fields")

    # Functions
    def clear_all_fields():
        return "", "", "", "", ""

    # Button Click Events
    Detailed_Info_Barcode_Buton.click(fn=scan_barcode_info, outputs=Detailed_Info_Barcode)
    Detailed_InfoOfIngredient_Barcode_Button.click(fn=scan_barcode_ingred, outputs=Detailed_InfoOfIngredient_Barcode)
    User_input_Button.click(fn=process_input, inputs=input_box, outputs=User_Input)
    upload_button.click(fn=process_and_summarize_ocr_image, inputs=file_input, outputs=ocr_output_box)
    clear_button.click(fn=clear_all_fields, outputs=[input_box, Detailed_Info_Barcode, Detailed_InfoOfIngredient_Barcode, User_Input, ocr_output_box])

demo.launch(share=True)


