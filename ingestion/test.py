import fitz

pdf = fitz.open("data/pdfs/std10/computer/compnew.pdf")

total_chars = 0

for page in pdf:
    total_chars += len(page.get_text())

print("Pages      :", len(pdf))
print("Characters :", total_chars)

pdf.close()