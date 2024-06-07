"""
PDF Text Summarization using LangChain
"""

from langchain import pipeline, LangChainForConditionalGeneration, LangChainTokenizer
import fitz  # PyMuPDF library
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def read_pdf(file_path):
    """
    Read text from a PDF file.

    Parameters:
    - file_path (str): The path to the PDF file.

    Returns:
    - str: The extracted text from the PDF.
    """
    text = ""
    with fitz.open(file_path) as pdf_document:
        num_pages = pdf_document.page_count
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            text += page.get_text()
    return text

def write_pdf(file_path, original_text, summary_text):
    """
    Write original text and summary to a new PDF file.

    Parameters:
    - file_path (str): The path to the output PDF file.
    - original_text (str): The original text to be included in the PDF.
    - summary_text (str): The summary text to be included in the PDF.
    """
    pdf_canvas = canvas.Canvas(file_path, pagesize=letter)
    pdf_canvas.drawString(100, 800, "Original Text:")
    pdf_canvas.drawString(100, 780, original_text)
    pdf_canvas.drawString(100, 760, "Summary:")
    pdf_canvas.drawString(100, 740, summary_text)
    pdf_canvas.save()

def generate_summary(input_text, model, tokenizer, max_length=1000, min_length=100):
    """
    Generate a summary using LangChain.

    Parameters:
    - input_text (str): The input text to be summarized.
    - model: LangChain model for conditional generation.
    - tokenizer: LangChain tokenizer.
    - max_length (int): Maximum length of the generated summary.
    - min_length (int): Minimum length of the generated summary.

    Returns:
    - str: The generated summary.
    """
    # Split the input text into chunks to avoid exceeding the maximum sequence length
    chunk_size = 1024  # Adjust as needed based on the model's maximum sequence length
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]

    # Generate summaries for each chunk
    summaries = []
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, early_stopping=False, num_beams=1)
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length)
        summaries.append(summary[0]['summary'])

    # Concatenate the summaries to obtain the final result
    final_summary = ' '.join(summaries)
    return final_summary

# Load a specific LangChain model and tokenizer
model_name = "facebook/langchain-large-cnn"
model = LangChainForConditionalGeneration.from_pretrained(model_name)
tokenizer = LangChainTokenizer.from_pretrained(model_name)

# Provide the path to the PDF file
input_pdf_path = "C:\\Users\\mzaid295\\input_document.pdf"

# Read input from a PDF file
input_text = read_pdf(input_pdf_path)

# Generate summary
output_summary = generate_summary(input_text, model, tokenizer)

# Print and write results to a new PDF file
print("Original Text:\n", input_text)
print("\nSummary:\n", output_summary)

output_pdf_path = "C:\\Users\\mzaid295\\output_summary.pdf"
write_pdf(output_pdf_path, input_text, output_summary)

'''================================== task2 =========================================
Title: Personalized Study Plan for [Student Name]
Objective: To create a comprehensive and tailored study plan for [Student Name] that takes into account their academic requirements, preferred learning styles, extracurricular activities, and personal objectives or challenges.
Introduction:
* Briefly introduce the student and provide any relevant background information, such as their academic level (e.g., high school, college), field of study, and current academic performance.
* Mention the purpose of the study plan and how it will be used to support the student's academic success.
Academic Requirements:
* List the student's current courses and academic requirements, including any specific exams, assignments, or projects that need to be completed.
* Provide information about the student's academic performance in each course, including their current grade level and any areas of concern.
Learning Styles:
* Identify the student's preferred learning styles, such as visual, auditory, or kinesthetic.
* Explain how the student's learning styles can be accommodated in the study plan, such as incorporating visual aids, audio recordings, or hands-on activities.
Extracurricular Activities:
* List the student's extracurricular activities, such as sports, clubs, or volunteer work.
* Explain how the student's extracurricular activities can be integrated into the study plan, such as scheduling study sessions around practice times or using club activities to reinforce academic concepts.
Personal Objectives or Challenges:
* Identify the student's personal objectives or challenges, such as preparing for a specific exam or overcoming a learning difficulty.
* Explain how the study plan can address these objectives or challenges, such as incorporating targeted practice exercises or seeking additional support from a tutor or mentor.
Study Plan:
* Provide a detailed study plan that takes into account the student's academic requirements, learning styles, extracurricular activities, and personal objectives or challenges.
* Include a schedule of study sessions, breaks, and review periods, as well as strategies for addressing specific academic challenges or objectives.
* Suggest resources or tools that can support the student's learning, such as textbooks, online resources, or productivity apps.
Conclusion:
* Summarize the key points of the study plan and emphasize the importance of tailoring the plan to the student's unique needs and aspirations.
* Encourage the student to take an active role in implementing the study plan and seeking support from teachers, tutors, or mentors as needed.
By structuring the prompt in this way, the AI can synthesize the diverse array of student-specific data to create a personalized study plan that addresses academic requirements, aligns with the student's preferred learning styles and extracurricular activities, and supports their unique objectives or challenges
'''