import pdfplumber
from markitdown import MarkItDown

md=MarkItDown()

result=md.convert("example.pdf")

print(result.markdown)