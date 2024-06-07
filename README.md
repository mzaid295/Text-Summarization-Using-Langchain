# PDF Text Summarization using LangChain

This script utilizes LangChain to summarize text extracted from a PDF file. Demonstrates how to summarize text from a PDF using the LangChain library for natural language processing. The code reads a PDF file, generates a summary of its content, and writes both the original text and the summary to a new PDF file.

# Requirements:

1. langchain
2. fitz (PyMuPDF library)
3. reportlab

# Installation:
pip install langchain fitz reportlab

#  Main Execution

**Load Model and Tokenizer:** Load a specific LangChain model and tokenizer.
**Read Input PDF:** Provide the path to the PDF file and read its content.
**Generate Summary:** Use the loaded model and tokenizer to generate a summary of the PDF text.
**Write Results to PDF:** Write both the original text and the generated summary to a new PDF file.

#  Example
Here is an example of how to use the provided functions:

1. Place the PDF file you want to summarize at C:\\Users\\mzaid295\\input_document.pdf.
2. Run the script.
3. The output summary will be printed on the console and saved in C:\\Users\\mzaid295\\output_summary.pdf.
By following these steps, you can summarize the content of a PDF and create a new PDF containing both the original text and its summary.
