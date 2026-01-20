# Automated BI-RADS-Oriented Radiology Report Generation

This repository contains a research prototype for automated radiology report generation in breast imaging using a Retrieval-Augmented Generation (RAG) pipeline and large language models (LLMs). The system generates BI-RADS-compliant English radiology reports from structured clinical findings, with a focus on terminology consistency and report standardization.

This project was developed as part of a university-level Research Project course and is intended for academic and experimental purposes only.

---

## Project Overview

Radiology reporting is a critical component of breast cancer diagnosis and follow-up. Although the Breast Imaging Reporting and Data System (BI-RADS) provides a standardized framework for terminology and assessment categories, report generation remains a largely manual and time-consuming task.

This project explores how natural language processing techniques and large language models can be used to support radiologists by transforming structured breast imaging findings into coherent and standardized radiology reports. To mitigate known risks of unconstrained text generation, the system employs a Retrieval-Augmented Generation approach, grounding report generation in authoritative BI-RADS guideline content.

---

## Objectives

- Generate BI-RADS-compliant radiology reports from structured breast imaging findings  
- Reduce linguistic variability while preserving standardized medical terminology  
- Explore retrieval-based grounding as an alternative to rigid rule-based systems  
- Provide a reproducible and explainable NLP pipeline for research purposes  

---

## System Architecture

The system follows a modular pipeline consisting of the following stages:

1. **Input Processing**  
   Structured clinical descriptors such as lesion type, shape, margin, and echo pattern are provided as system input.

2. **Knowledge Retrieval**  
   Relevant BI-RADS guideline excerpts are retrieved from a vector database using semantic similarity search.

3. **Report Generation**  
   A large language model generates the final report conditioned on both the structured findings and the retrieved guideline context.

4. **Post-processing and Review**  
   Generated report sections are extracted and presented for optional manual review and editing.

---

## Data Description

- No real patient data are used in this project.  
- Evaluation is performed on synthetic structured cases derived from the ACR BI-RADS Atlas (5th Edition).  
- Synthetic cases cover a range of BI-RADS categories and lesion characteristics to support controlled testing.

---

## Technologies Used

- Python 3.9  
- Streamlit (user interface)  
- LangChain (LLM orchestration)  
- ChromaDB (vector storage and retrieval)  
- OpenAI GPT-4o (text generation)  
- OpenAI Embeddings (semantic retrieval)  
- Regular expressions (post-processing)  

---

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/berilctl/Automated_Radiology_Report_Generator.git
cd Automated_Radiology_Report_Generator
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Initialize the vector database:

```bash
python ingest.py
```

Run the application:

```bash
streamlit run app.py
```

---

## Usage

1. Start the Streamlit application using the command above
2. Enter structured clinical findings in the input form
3. The system will retrieve relevant BI-RADS guidelines and generate a compliant report
4. Review and edit the generated report as needed

---

## Project Structure

```
├── app.py                 # Streamlit application
├── ingest.py              # Vector database initialization
├── requirements.txt       # Python dependencies
├── data.csv              # Sample structured data
├── docs/                 # Documentation
│   └── birads_mini_guide.txt
└── README.md             # This file
```

---

## License

This project is intended for academic and research purposes only.

---

## Disclaimer

This system is a research prototype and should not be used for actual clinical decision-making. Always consult qualified medical professionals for medical advice and diagnosis.
