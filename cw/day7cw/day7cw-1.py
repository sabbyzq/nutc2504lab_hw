import pdfplumber

with pdfplumber.open("example.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())
