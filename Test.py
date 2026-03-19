import streamlit as st
import pytesseract
from PIL import Image
import ollama
from pdf2image import convert_from_bytes

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("OCR + PDF + Local LLM Demo")

uploaded_file = st.file_uploader(
    "Upload image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

extracted_text = ""

if uploaded_file:

    if uploaded_file.type == "application/pdf":

        st.subheader("Processing PDF")

        images = convert_from_bytes(uploaded_file.read())

        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_container_width=True)

            text = pytesseract.image_to_string(img)
            extracted_text += text + "\n"

    else:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        extracted_text = pytesseract.image_to_string(image)

    st.subheader("Extracted Text")
    st.text_area("OCR Output", extracted_text, height=200)

    if st.button("Analyze with LLM"):

        response = ollama.chat(
            model="phi3:latest",
            messages=[
                {
                    "role": "user",
                    "content": f"""
This text was extracted from OCR.

Clean it and summarize the document.

Text:
{extracted_text}
"""
                }
            ]
        )

        st.subheader("LLM Response")
        st.write(response["message"]["content"])