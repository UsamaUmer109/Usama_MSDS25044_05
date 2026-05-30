import sys
from PyPDF2 import PdfReader

if len(sys.argv) < 2:
    print("Usage: extract_pdf_text.py input.pdf [output.txt]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else input_path + ".txt"

reader = PdfReader(input_path)
texts = []
for p in reader.pages:
    t = p.extract_text()
    if t:
        texts.append(t)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(texts))

print("Wrote:", output_path)
