import os
import requests
import pdfplumber

def convert_pdf_to_text_pdfplumber(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber, including page markers.
    """
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text_content += f"\n[PAGE {page_num}]\n" + text + "\n"
        return text_content
    except Exception as e:
        print(f"An error occurred during pdfplumber conversion of {pdf_path}: {e}")
        return None

# Path to your Google Drive's "My Drive"
gdrive_path = os.getcwd()

folders = ['sample_docs']

pdf_filenames = []
text_filenames = []

for folder in folders:
    folder_path = os.path.join(gdrive_path, folder)

    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for item in files:
                if item.endswith(".pdf"):
                    pdf_filenames.append(os.path.join(root, item))
                    text_filenames.append(item[:-4] + ".txt")
    else:
        print(f"The folder '{folder_path}' does not exist in your directory.")

print("PDF Filenames:", pdf_filenames)
print("Text Filenames:", text_filenames)
print("Number of PDF files found:", len(pdf_filenames))

for i in range(len(pdf_filenames)):
    pdf_filename = pdf_filenames[i]
    text_filename = text_filenames[i]

    try:
        # Convert PDF to text using pdfplumber
        print(f"Converting {pdf_filename} to {text_filename} using pdfplumber...")
        text_content = convert_pdf_to_text_pdfplumber(pdf_filename)

        if text_content is not None:
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"Successfully converted and saved to {text_filename}")
        else:
            print(f"Failed to convert {pdf_filename} to text.")

    except Exception as e:
        print(f"An unexpected error occurred for {pdf_filename}: {e}")
