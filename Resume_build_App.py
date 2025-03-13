import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_groq import ChatGroq  # Correct import for ChatGroq
from langchain.prompts import PromptTemplate
from fpdf import FPDF

load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_resume(name: str, email: str, phone: str, skills: str, experience: str, education: str) -> str:
    """
    Generates a resume using Groq Llama model.
    """
    prompt_template: str = f"""
    Create a professional resume based on the following details:
    
    Name: {name}
    Email: {email}
    Phone: {phone}
    Skills: {skills}
    Experience: {experience}
    Education: {education}
    
    Format it properly for a professional resume.
    """
    
    prompt_gen = PromptTemplate(template=prompt_template, input_variables=["name", "email", "phone", "skills", "experience", "education"])
    llm: Any = ChatGroq(model_name="llama-3.2-11b-vision-preview", temperature=0.7)  # Updated model
    resume_chain: Any = LLMChain(llm=llm, prompt=prompt_gen, verbose=True)
    generated_resume: str = resume_chain.predict(name=name, email=email, phone=phone, skills=skills, experience=experience, education=education)
    return generated_resume

def save_resume_as_pdf(resume_text: str, filename: str) -> None:
    """
    Saves the generated resume as a PDF file.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, resume_text)
    pdf.output(filename)

def main() -> None:
    """
    Main function to build a Resume Web App using Streamlit.
    """
    st.set_page_config(page_title="AI Resume Builder", page_icon="📄")
    
    with st.sidebar:
        st.image("PragyanAI_Transperent_github.png")
        st.write("AI App created for educational purposes.")
    
    st.header("AI Resume Builder")
    
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    skills = st.text_area("Skills (comma-separated)")
    experience = st.text_area("Work Experience")
    education = st.text_area("Education")
    
    if st.button("Generate Resume"):
        if name and email and phone and skills and experience and education:
            resume_text = generate_resume(name, email, phone, skills, experience, education)
            pdf_filename = "resume.pdf"
            save_resume_as_pdf(resume_text, pdf_filename)
            st.success("Resume Generated Successfully!")
            with open(pdf_filename, "rb") as pdf_file:
                st.download_button("Download Resume as PDF", pdf_file, file_name=pdf_filename, mime="application/pdf")
        else:
            st.error("Please fill in all fields before generating the resume.")

if __name__ == "__main__":
    main()
