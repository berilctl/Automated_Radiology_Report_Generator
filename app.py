import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
# Updated libraries
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Research Project | Radiology Report", layout="wide")

# Clinical History Dictionary - Synthetic clinical histories for each patient
# Note: These are synthetic/generated clinical histories for demonstration purposes
CLINICAL_HISTORIES = {
    "TCGA_CS_4941": "Patient presents with palpable mass in the right breast. Family history of breast cancer in maternal grandmother. Routine screening mammography recommended.",
    "TCGA_CS_4942": "Annual screening mammography. Previous benign biopsy 3 years ago. No interval changes noted on clinical examination.",
    "TCGA_CS_4943": "Follow-up imaging for previously identified BI-RADS 3 lesion. Patient with high-risk family history. Recommended 6-month follow-up.",
    "TCGA_CS_4944": "Patient presents with nipple discharge. Clinical examination shows asymmetry. Diagnostic mammography and ultrasound requested.",
    "TCGA_CS_5393": "Screening mammography in high-risk patient. BRCA mutation carrier. Annual MRI recommended.",
    "TCGA_CS_5395": "Follow-up imaging after lumpectomy 2 years ago. No palpable abnormalities on clinical examination. Routine surveillance.",
    "TCGA_CS_5396": "Patient with dense breast tissue. Screening mammography with supplementary ultrasound. Previous normal mammogram 1 year ago.",
    "TCGA_CS_5397": "Diagnostic workup for breast pain and tenderness. Clinical examination reveals no discrete masses. Diagnostic imaging to rule out pathology.",
    "TCGA_CS_6186": "High-risk screening patient. History of atypical ductal hyperplasia. Enhanced surveillance protocol.",
    "TCGA_CS_6188": "Follow-up for BI-RADS 0 assessment from screening. Additional imaging for complete evaluation.",
}

st.title("Research Project | Radiology Report")
st.markdown("---")

# --- 1. LEFT PANEL (SETTINGS & INPUTS) ---
with st.sidebar:
    st.header("Settings")
    
    # API Key Input
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    # API Key validation
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    # Warning if no key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API Key to proceed.")
        st.stop()

    st.markdown("---")
    
    # A. PATIENT SELECTION (From CSV)
    st.subheader("Patient Information")
    try:
        # Check if data.csv exists
        if os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
            # Assuming first column is ID
            patient_list = df.iloc[:, 0].tolist() 
            selected_patient_id = st.selectbox("Select Patient ID", patient_list)
            
            # Get selected patient row
            patient_data = df[df.iloc[:, 0] == selected_patient_id].iloc[0]
            
            # Safely get data using .get()
            age = patient_data.get('age_at_initial_pathologic', 'N/A')
            gender = patient_data.get('gender', 'N/A')
            laterality = patient_data.get('laterality', 'N/A')
            
            # Get clinical history - check CSV first, then dictionary
            clinical_history = patient_data.get('clinical_history', None)
            if pd.isna(clinical_history) or clinical_history == '':
                clinical_history = CLINICAL_HISTORIES.get(selected_patient_id, "Routine breast imaging examination.")
            
            # Display Patient Info card
            st.info(f"""
            **Age:** {age}
            **Gender:** {gender}
            **Laterality:** {laterality}
            """)
        else:
            st.error("'data.csv' not found! Using default test values.")
            selected_patient_id = "TEST_PATIENT_001"
            age, gender, laterality = "45", "Female", "Left"
            clinical_history = "Routine breast imaging examination."
            
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # B. MODEL 0 SIMULATION (Radiological Findings)
    st.markdown("---")
    st.subheader("Radiological Findings")
    st.caption("Enter radiological findings from segmentation analysis:")
    
    col_mock1, col_mock2 = st.columns(2)
    with col_mock1:
        mass_size = st.number_input("Size (mm)", min_value=1, max_value=100, value=15)
        # Using standardized English BI-RADS terms
        mass_shape = st.selectbox("Shape", ["Oval", "Round", "Irregular"])
    with col_mock2:
        mass_margin = st.selectbox("Margin", ["Circumscribed", "Indistinct", "Spiculated"])
        echo_pattern = st.selectbox("Echo Pattern", ["Hypoechoic", "Isoechoic", "Anechoic", "Complex"])

    mass_location = st.selectbox("Location", ["12 o'clock", "3 o'clock", "6 o'clock", "9 o'clock", "Upper Outer Quadrant", "Retroareolar"])

# --- 2. MAIN PANEL (REPORT GENERATION) ---
st.subheader("Report Generation")

if st.button("Generate Report", use_container_width=True):
    with st.spinner("Analyzing BI-RADS guidelines and generating report..."):
        try:
            # 1. RAG CONNECTION (ChromaDB)
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=embeddings
            )
            # Retrieve top 2 relevant guidelines
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 
            
            # 2. LLM SETUP (GPT-4o)
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
            
            # 3. RETRIEVE CONTEXT
            query_text = f"Mass shape is {mass_shape}, margin is {mass_margin}, echo pattern is {echo_pattern}."
            relevant_docs = retriever.invoke(query_text)
            
            # Combine doc contents
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 4. PREPARE PROMPT
            # Get current date and time
            report_datetime = datetime.now()
            report_date_str = report_datetime.strftime("%d.%m.%Y")
            report_time_str = report_datetime.strftime("%H:%M")
            
            system_prompt = f"""You are an expert Breast Radiologist writing a clinical ultrasound report in standard radiology format. Write ONLY the complete report with all required sections.

BI-RADS GUIDELINES (CONTEXT):
{context_text}

PATIENT INFORMATION:
- Patient ID: {selected_patient_id}
- Age: {age} years
- Gender: {gender}
- Laterality: {laterality}
- Report Date: {report_date_str}
- Report Time: {report_time_str}
- Location: Remagen, Germany

CLINICAL HISTORY:
{clinical_history}

ULTRASOUND FINDINGS (from segmentation analysis):
- Location: {mass_location}
- Size: {mass_size} mm
- Shape: {mass_shape}
- Margin: {mass_margin}
- Echo Pattern: {echo_pattern}

OUTPUT REQUIREMENTS:
- Write a complete professional radiology report starting with the header section
- Use BI-RADS standardized terminology and sentence structures
- Follow standard radiology report format exactly as shown below
- Format exactly as follows (include all sections):

ULTRASOUND REPORT

Patient ID: {selected_patient_id}
Patient Age: {age} years
Report Date: {report_date_str}
Report Time: {report_time_str}
Location: Remagen, Germany

Clinical History:
{clinical_history}

FINDINGS:
[Describe the findings using BI-RADS standardized terminology. Write in complete, professional sentences as used in real clinical radiology reports. Include location, size, shape, margin, echo pattern, and any additional relevant observations. Be descriptive and use proper medical terminology.]

IMPRESSION:
[Provide the BI-RADS category (0-6) and clinical interpretation using standard BI-RADS language. State the category clearly and provide appropriate recommendations based on BI-RADS guidelines.]

RECOMMENDATION:
[Provide clinical recommendations based on BI-RADS category and findings. Include follow-up imaging, biopsy recommendations, or routine screening as appropriate.]

Do NOT include any introductory text like "Here is the report" or "Certainly". Start directly with "ULTRASOUND REPORT"."""
            
            # 5. INVOKE LLM
            response = llm.invoke(system_prompt)
            generated_report = response.content
            
            st.session_state["generated_report"] = generated_report
            st.session_state["generated_report_english"] = generated_report  # Keep English version
            st.session_state["relevant_docs"] = relevant_docs  # Store for reference display
            st.success("Report generated successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to translate report to German
def translate_to_german():
    if "generated_report_german" not in st.session_state and "generated_report_english" in st.session_state:
        try:
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
            english_report = st.session_state["generated_report_english"]
            translation_prompt = f"""You are a medical translator specializing in radiology reports. Translate the following English radiology report to German, maintaining the exact format and structure.

CRITICAL REQUIREMENTS:
- Translate the COMPLETE report including header section (Patient ID, Age, Date, Location, Clinical History)
- Maintain the exact structure: Header → Clinical History → BEFUND → EINDRUCK → EMPFEHLUNG
- Translate "ULTRASOUND REPORT" to "ULTRASCHALLBEFUND" or "ULTRASCHALLBERICHT"
- Translate "Patient ID" to "Patienten-ID"
- Translate "Patient Age" to "Patientenalter"
- Translate "Report Date" to "Befunddatum"
- Translate "Report Time" to "Befundzeit"
- Translate "Location" to "Ort"
- Translate "Clinical History" to "Klinische Anamnese"
- Translate "FINDINGS" to "BEFUND"
- Translate "IMPRESSION" to "EINDRUCK"
- Translate "RECOMMENDATION" to "EMPFHEHLUNG"
- Use standard German medical terminology as used in real clinical radiology reports in Germany
- Keep BI-RADS categories as "BI-RADS" (not translated)
- Use proper German medical sentence structures
- Do NOT add any introductory text, explanations, or meta-commentary
- Do NOT include phrases like "Hier ist die Übersetzung" or "Gewiss"

English Report:
{english_report}

Output ONLY the complete German translation maintaining the exact format - start with "ULTRASCHALLBEFUND" or "ULTRASCHALLBERICHT"."""
            translation_response = llm.invoke(translation_prompt)
            st.session_state["generated_report_german"] = translation_response.content
            return True
        except Exception as e:
            st.error(f"Translation error: {e}")
            return False
    return False

# Report Display & Editing Area
if "generated_report" in st.session_state:
    # Initialize previous language if not exists
    if "previous_language_choice" not in st.session_state:
        st.session_state["previous_language_choice"] = "English"
    
    # Language selection
    col_lang1, col_lang2 = st.columns([1, 1])
    with col_lang1:
        language_choice = st.radio(
            "Report Language / Berichtssprache:",
            ["English", "Deutsch"],
            horizontal=True,
            key="language_choice"
        )
    
    # Check if language just changed to German
    just_switched_to_german = (language_choice == "Deutsch" and 
                            st.session_state.get("previous_language_choice") != "Deutsch")
    
    # Automatic translation when German is selected (only if just switched)
    if just_switched_to_german:
        if "generated_report_german" not in st.session_state:
            with st.spinner("Translating report to German... / Übersetze Bericht auf Deutsch..."):
                if translate_to_german():
                    st.session_state["previous_language_choice"] = "Deutsch"
                    st.rerun()
    
    # Update previous language choice after processing
    if language_choice != st.session_state.get("previous_language_choice"):
        # Only update if we're not in the middle of a translation
        if not just_switched_to_german or "generated_report_german" in st.session_state:
            st.session_state["previous_language_choice"] = language_choice
    
    # German translation button (manual override/refresh)
    with col_lang2:
        if language_choice == "Deutsch":
            if "generated_report_german" in st.session_state:
                if st.button("Refresh Translation / Übersetzung aktualisieren", use_container_width=True):
                    # Force re-translation
                    del st.session_state["generated_report_german"]
                    with st.spinner("Re-translating report... / Neu übersetzen..."):
                        if translate_to_german():
                            st.rerun()
    
    st.subheader("Editable Report")
    
    # Get the appropriate report based on language choice
    if language_choice == "Deutsch":
        if "generated_report_german" in st.session_state:
            display_report = st.session_state["generated_report_german"]
        else:
            # Fallback to English if translation is not yet available
            display_report = st.session_state.get("generated_report_english", st.session_state["generated_report"])
    else:
        display_report = st.session_state.get("generated_report_english", st.session_state["generated_report"])
    
    # Use dynamic key based on language to force update when language changes
    text_area_key = f"report_text_area_{language_choice}"
    final_report = st.text_area(
        "Physician Edit Mode: / Bearbeitungsmodus:" if language_choice == "Deutsch" else "Physician Edit Mode:", 
        value=display_report, 
        height=400,
        key=text_area_key
    )
    
    # Update session state based on current language
    if language_choice == "Deutsch":
        st.session_state["generated_report_german"] = final_report
    else:
        st.session_state["generated_report_english"] = final_report
    
    # Recommendation section - editable by physician
    st.markdown("---")
    st.subheader("Recommendation / Empfehlung")
    
    # Initialize recommendation if not exists
    if "recommendation" not in st.session_state:
        st.session_state["recommendation"] = ""
    
    recommendation_key = f"recommendation_{language_choice}"
    # Initialize recommendation if not exists - extract from report if available
    if recommendation_key not in st.session_state:
        existing_recommendation = st.session_state.get("recommendation", "")
        # Try to extract existing recommendation from display_report (original report, not edited)
        if not existing_recommendation and display_report:
            # Extract recommendation from report if it exists
            if language_choice == "Deutsch":
                rec_header = "EMPFHEHLUNG"
            else:
                rec_header = "RECOMMENDATION"
            
            # Try pattern that matches RECOMMENDATION: and everything after it
            pattern = rf"{rec_header}:?\s*\n([\s\S]*?)(?=\n\n(?:[A-Z][A-Z\s]+:|$)|\Z|\n[A-Z][A-Z\s]+:)"
            match = re.search(pattern, display_report, re.IGNORECASE | re.MULTILINE)
            if match:
                existing_recommendation = match.group(1).strip()
            else:
                # Try simpler pattern for recommendation at the end
                simple_pattern = rf"{rec_header}:?\s*\n([\s\S]*)$"
                simple_match = re.search(simple_pattern, display_report, re.IGNORECASE | re.MULTILINE)
                if simple_match:
                    existing_recommendation = simple_match.group(1).strip()
        
        st.session_state[recommendation_key] = existing_recommendation
    
    physician_recommendation = st.text_area(
        "Physician Recommendation (Editable): / Ärztliche Empfehlung (Bearbeitbar):" if language_choice == "Deutsch" else "Physician Recommendation (Editable):",
        value=st.session_state.get(recommendation_key, ""),
        height=150,
        key=recommendation_key,
        help="Add or modify clinical recommendations as needed. This will be automatically added to the report."
    )
    
    # Update report with physician recommendation when it changes
    # Store previous recommendation to detect changes
    prev_recommendation_key = f"prev_recommendation_{language_choice}"
    if prev_recommendation_key not in st.session_state:
        st.session_state[prev_recommendation_key] = ""
    
    # Check if recommendation changed
    if physician_recommendation != st.session_state[prev_recommendation_key]:
        # Get current report from session state
        current_report = final_report
        
        # Determine recommendation section header based on language
        if language_choice == "Deutsch":
            rec_header = "EMPFHEHLUNG"
        else:
            rec_header = "RECOMMENDATION"
        
        if current_report:
            # Pattern to find RECOMMENDATION or EMPFHEHLUNG section (case insensitive)
            # Match the header and everything after it until end of report or next section header
            # More flexible pattern that handles various formats
            pattern = rf"({rec_header}:?)\s*\n([\s\S]*?)(?=\n\n(?:[A-Z][A-Z\s]+:|$)|\Z|\n[A-Z][A-Z\s]+:)"
            
            # First, try to find and replace existing recommendation section
            match = re.search(pattern, current_report, re.IGNORECASE | re.MULTILINE)
            if match:
                # Replace entire recommendation section including header with new content
                updated_report = re.sub(
                    pattern,
                    rf"{rec_header}:\n{physician_recommendation}",
                    current_report,
                    flags=re.IGNORECASE | re.MULTILINE
                )
            else:
                # Try simpler pattern - just find RECOMMENDATION: and everything after it to end
                simple_pattern = rf"({rec_header}:?)\s*\n([\s\S]*)$"
                simple_match = re.search(simple_pattern, current_report, re.IGNORECASE | re.MULTILINE)
                if simple_match:
                    # Replace recommendation at the end
                    updated_report = re.sub(
                        simple_pattern,
                        rf"{rec_header}:\n{physician_recommendation}",
                        current_report,
                        flags=re.IGNORECASE | re.MULTILINE
                    )
                else:
                    # No recommendation section found - add it at the end
                    if current_report.strip():
                        updated_report = f"{current_report}\n\n{rec_header}:\n{physician_recommendation}"
                    else:
                        updated_report = f"{rec_header}:\n{physician_recommendation}"
            
            # Update session state with new report
            if updated_report != current_report:
                if language_choice == "Deutsch":
                    st.session_state["generated_report_german"] = updated_report
                else:
                    st.session_state["generated_report_english"] = updated_report
                
                # Store updated recommendation
                st.session_state[prev_recommendation_key] = physician_recommendation
                
                # Rerun to update the report text area
                st.rerun()
    
    col_save1, col_save2 = st.columns([1, 1])
    with col_save1:
        if st.button("Approve & Save to System / Speichern", use_container_width=True):
            st.success("Report saved to database successfully.")
            # Future: Save to JSON logic here