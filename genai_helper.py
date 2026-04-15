import google.generativeai as genai

# Use the recommended fast text model
MODEL_NAME = "gemini-1.5-flash"

def configure_gemini(api_key: str) -> None:
    """Configures the Gemini API client with the given key."""
    if api_key:
        genai.configure(api_key=api_key)

def check_configured() -> bool:
    # A simple way to check if an API key is currently in the environment or configured
    # In practice, we will check if the user entered it in Streamlit.
    return True

def generate_candidate_analysis(resume_text: str, jd: str) -> str:
    """
    Generates a deep analysis of the candidate's resume against the job description.
    """
    prompt = f"""
    You are an expert technical recruiter analyzing a candidate's resume for a specific job description.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Provide a comprehensive analysis including:
    1. Recommendation (Hire / Hold / Reject) based on the match.
    2. Deep Gap Analysis: What specific skills or experiences are missing or weakly supported?
    3. Strengths: What are the best matching parts of this resume?
    4. Red Flags: Are there any concerns (e.g., job hopping, missing core requirements)?
    
    Be objective, precise, and concise. Use Markdown formatting.
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def generate_interview_questions(resume_text: str, jd: str) -> str:
    """
    Generates tailored interview questions for the candidate based on their resume and the JD.
    """
    prompt = f"""
    You are an expert technical interviewer preparing to interview a candidate.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Based on the gaps and strengths of the candidate's resume relative to the job description, 
    generate 5 specific, tailored interview questions. 
    Focus on verifying claims made in the resume that are highly relevant to the JD, 
    and probing any areas where the candidate's experience seems thin compared to the requirements.
    
    Format as a numbered Markdown list.
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating interview questions: {str(e)}"

def generate_email_draft(candidate_name: str, resume_text: str, jd: str, status: str) -> str:
    """
    Generates a draft email (acceptance or rejection) for the candidate.
    """
    if status.lower() == "accept":
        intent = "invite them to an interview, mentioning a couple of specific strengths from their resume that excited the team."
    else:
        intent = "politely reject them for the position, providing one specific piece of constructive feedback based on the main gap between their resume and the JD."
        
    prompt = f"""
    You are a technical recruiter drafting an email to a candidate named {candidate_name}.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Your goal is to {intent}
    
    Draft the email. Make it professional, empathetic, and concise.
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating email draft: {str(e)}"
