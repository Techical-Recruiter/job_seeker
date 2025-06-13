import streamlit as st
import PyPDF2
from docx import Document
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="CV Ranker - Job Seeker")

def input_text(uploaded_file):
    file_name = uploaded_file.name.lower()
    text = ""
    try:
        if file_name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in reader.pages)
        elif file_name.endswith((".doc", ".docx")):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""
    return text

def extract_json_from_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else None
        except:
            return None

def display_job_seeker_results(data):
    st.subheader("Detailed Analysis Results")

    if isinstance(data, dict):
        st.write(f"**Overall JD Match Score:** {data.get('##JD Match', 'N/A')}")

        st.subheader("Qualifications Breakdown", divider="blue")
        quals = data.get("##Qualifications Analysis", {})

        st.write(f"**Experience Match:** {quals.get('Experience Comparison', 'N/A')}")
        st.write(f"**Education Match:** {quals.get('Education Match', 'N/A')}")

        st.write("**Strengths:**")
        if strengths := quals.get("Strengths", []):
            for strength in strengths:
                st.success(f"✓ {strength}")
        else:
            st.write("No key strengths identified")

        st.write("**Skill Gaps:**")
        if gaps := quals.get("Skill Gaps", []):
            for gap in gaps:
                st.error(f"✗ {gap}")
        else:
            st.info("No major skill gaps identified")

        st.subheader("Career Improvement Advice", divider="green")
        improvements = data.get("##Improvement Suggestions", {})

        st.write("**Key Areas Needing Improvement:**")
        if areas := improvements.get("Key Areas", []):
            for area in areas:
                st.warning(f"⚠ {area}")
        else:
            st.info("No major improvement areas identified")

        st.write("**Actionable Advice:**")
        if advice := improvements.get("Actionable Advice", []):
            for item in advice:
                st.info(f"• {item}")
        else:
            st.write("No specific advice available")

        st.write(f"**Overall Career Fit:** {improvements.get('Career Fit', 'N/A')}")

        st.subheader("Keyword Analysis", divider="orange")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matching Keywords:**")
            if matching := data.get("##Matching Keywords", []):
                st.table(matching)
            else:
                st.write("None found")

        with col2:
            st.write("**Missing Keywords:**")
            if missing := data.get("##Missing Keywords", []):
                st.table(missing)
            else:
                st.write("None found")

        st.subheader("Career Counselor's Summary", divider="blue")
        st.write(data.get("##Profile Summary", "No summary available"))

    else:
        st.error("Could not parse response. Raw output:")
        st.code(data)

async def analyze_resume_job_seeker(uploaded_file, jd):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        agent = Agent(
            name="ATS Agent",
            instructions="""
            You are a career counselor analyzing a resume against a job description. Provide detailed feedback in this EXACT JSON format:

            {
                "##JD Match": "X%",
                "##Matching Keywords": ["keyword1", "keyword2"],
                "##Missing Keywords": ["keyword3", "keyword4"],
                "##Qualifications Analysis": {
                    "Experience Comparison": "JD requires X years, candidate has Y years (Overqualified/Underqualified/Good match)",
                    "Education Match": "How well the education matches (Excellent/Good/Fair/Poor)",
                    "Skill Gaps": ["List of important skills missing"],
                    "Strengths": ["List of strong matching skills"]
                },
                "##Improvement Suggestions": {
                    "Key Areas": ["List 2-3 key areas needing improvement"],
                    "Actionable Advice": ["Specific actionable advice for each area"],
                    "Career Fit": "How well the candidate fits this role (Excellent/Good/Fair/Poor)"
                },
                "##Profile Summary": "Concise 50-word summary of fit and key recommendations"
            }

            Analyze thoroughly and provide:
            1. Precise comparison of required vs actual experience years
            2. Detailed education/qualification matching
            3. Specific skill gaps and strengths
            4. Actionable improvement advice
            5. Honest assessment of over/under qualification
            6. Clear career fit assessment
            7. Act like you are talking straight to the job seeker

            Keep your response short and effective as much as possible. Be brutally honest but constructive. Focus on helping the candidate improve.
            """,
            model=model,
        )

        st.subheader(f"Analysis for Resume: {uploaded_file.name}")
        text = input_text(uploaded_file)
        if not text:
            return False

        resume_input = f"Resume Content:\n{text}\n\nJob Description:\n{jd}"
        result = Runner.run_streamed(starting_agent=agent, input=resume_input)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = extract_json_from_response(full_response)
        if response_json:
            display_job_seeker_results(response_json)
            return True
        else:
            st.error("Could not parse response from AI. Please try again.")
            return False
    except Exception as e:
        st.error(f"Server error: {str(e)}. Please try again later.")
        return False

def display_pakistan_recruitment_promo():
    st.markdown("---") 
    st.markdown(
        """
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
            <h3 style="color:#0056b3;">Elevate Your Career with PakistanRecruitment!</h3>
            <p>
                We hope this analysis helps you on your job-seeking journey.
                For more career opportunities, expert advice, and to connect with top employers,
                join the PakistanRecruitment community!
            </p>
            <p>
                🌐 <a href="https://pakistanrecruitment.com/" target="_blank" style="color:#007bff; text-decoration:none; font-weight:bold;">Visit Our Website</a>
            </p>
            <p>
                📱 <a href="https://whatsapp.com/channel/0029Vawltbj7z4kbLbGxSY35" target="_blank" style="color:#25D366; text-decoration:none; font-weight:bold;">Join Our WhatsApp Broadcast Channel</a>
            </p>
            <p>
                🔗 <a href="https://www.linkedin.com/company/pakistanrecruitment/posts/?feedView=all" target="_blank" style="color:#0A66C2; text-decoration:none; font-weight:bold;">Connect with us on LinkedIn</a>
            </p>
            <p style="font-size:0.9em; color:#6c757d;">
                *Your gateway to success in Pakistan's job market.*
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def job_seeker_app():
    st.title("CV Analyzer & Career Coach")
    st.markdown("**Powered by PakistanRecruitment**", unsafe_allow_html=True)
    st.write("Welcome!")

    st.header("Match your CV with any Job Post in seconds.", divider="grey")
    st.markdown('''
        **How to use:**
        1. Upload your resume in PDF, MS Word (.doc, .docx) format.
        2. Paste the job description of role you want to apply for and see your compatibility with the job.
        3. Get a clear match score, see what's missing in your resume, and receive career advice to improve your chances for AI-related roles.
        4. Click Submit to analyze.
    ''')

    upload_file = st.file_uploader("Upload your resume", type=["pdf", "doc", "docx"],
                                    help="Please upload one PDF, MS Word (.doc, .docx) file",
                                    accept_multiple_files=False)
    jd = st.text_area("Paste job description", height=200)
    submit = st.button("Submit")

    if submit:
        if not upload_file:
            st.error("Please upload a resume.")
        elif not jd.strip():
            st.error("Please provide a job description to analyze the resume.")
        else:
            asyncio.run(analyze_resume_job_seeker(upload_file, jd))

    # Call the promotional section at the very end
    display_pakistan_recruitment_promo()

if __name__ == "__main__":
    job_seeker_app()
