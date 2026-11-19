from pypdf import PdfReader

reader = PdfReader("/Users/modithadassanayake/Desktop/AI_Class/Projects/Assignment_1/BMI Category Guidelines & Regional Dietary Adaptation.pdf")
number_of_pages = len(reader.pages)

text = ""
for i in range(0,number_of_pages):
    # temp = reader.pages[i]
    # text = text + "\n\n" +temp.extract_text()
    text = text + "\n\n" + reader.pages[i].extract_text()
    

# page = reader.pages[0]
# text = page.extract_text()
print(text)