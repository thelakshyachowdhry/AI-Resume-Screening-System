from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st

from evaluator import evaluate_model
from parser import extract_text_from_pdf
from preprocess import preprocess_text
from similarity import (
    compute_bert_similarity,
    compute_job_resume_similarity,
    compute_resume_to_resume_similarity,
)
from genai_helper import (
    configure_gemini,
    generate_candidate_analysis,
    generate_interview_questions,
    generate_email_draft,
)
from skills import analyze_skill_match, skill_sets_to_strings
from utils import (
    MODEL_DIR,
    ScoreBreakdown,
    compute_final_score,
    dataframe_to_csv_download,
    log_event,
    safe_load_joblib,
    setup_logging,
)


MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def load_model_and_vectorizer():
    clf = safe_load_joblib(MODEL_PATH)
    vec = safe_load_joblib(VECTORIZER_PATH)
    return clf, vec


def summarize_resume(text: str, max_chars: int = 300) -> str:
    if not text:
        return "No text could be extracted from this resume."
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@st.cache_resource(show_spinner=False)
def load_summarizer_pipeline():
    """
    Lazily load and cache transformer summarizer model.
    """
    from transformers import pipeline

    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
    )


def summarize_resume_ai(text: str, max_input_chars: int = 1800) -> Tuple[str, str | None]:
    """
    Optional AI summary. Falls back to basic summary if unavailable.
    """
    if not text:
        return "No text could be extracted from this resume.", None

    try:
        summarizer = load_summarizer_pipeline()
        clipped = " ".join(text.split())[:max_input_chars]
        if len(clipped.split()) < 40:
            return summarize_resume(text), None

        output = summarizer(
            clipped,
            max_length=90,
            min_length=25,
            do_sample=False,
        )
        if output and "summary_text" in output[0]:
            return output[0]["summary_text"].strip(), None
        return summarize_resume(text), "Unexpected summarization output format."
    except Exception as e:
        return summarize_resume(text), str(e)


def explain_candidate(
    name: str,
    skills_found: List[str],
    missing_skills: List[str],
    tfidf_score_pct: float,
    bert_score_pct: float,
    final_score_pct: float,
) -> str:
    reasons: List[str] = []
    if tfidf_score_pct >= 70:
        reasons.append("high textual similarity to the job description")
    elif tfidf_score_pct >= 40:
        reasons.append("moderate similarity to the job description")
    else:
        reasons.append("low similarity to the job description")

    if bert_score_pct >= 70:
        reasons.append("strong semantic alignment captured by BERT")
    elif bert_score_pct >= 40:
        reasons.append("moderate semantic alignment captured by BERT")
    else:
        reasons.append("limited semantic alignment captured by BERT")

    if missing_skills:
        gap_phrase = f"lacks some important skills such as {', '.join(missing_skills[:3])}"
    else:
        gap_phrase = "covers almost all of the requested skills"

    return (
        f"{name} was scored based on {', '.join(reasons)}, and {gap_phrase}. "
        f"Final score: {final_score_pct:.1f}."
    )


def main() -> None:
    setup_logging()
    st.set_page_config(
        page_title="AI Resume Screening System",
        layout="wide",
    )

    st.title("AI Resume Screening System")
    st.write(
        "Upload multiple resumes and provide a job description. "
        "The system ranks candidates using TF-IDF similarity, BERT semantic similarity, "
        "skill matching, and a supervised machine learning model (Logistic Regression)."
    )

    with st.sidebar:
        st.header("Job Configuration")
        job_role = st.selectbox(
            "Job Role (optional preset)",
            ["Custom", "Data Scientist", "AI/ML Engineer", "Data Analyst"],
            index=0,
        )

        if job_role == "Data Scientist":
            default_jd = (
                "Looking for a data scientist with strong Python, machine learning, and SQL "
                "experience. Experience with scikit-learn and data visualization is a plus."
            )
        elif job_role == "AI/ML Engineer":
            default_jd = (
                "AI/ML engineer with deep learning experience using TensorFlow or PyTorch, "
                "strong Python, and experience deploying models to production."
            )
        elif job_role == "Data Analyst":
            default_jd = (
                "Data analyst with strong SQL, Excel, dashboarding (Tableau/Power BI), "
                "and basic statistics knowledge."
            )
        else:
            default_jd = ""

        job_description = st.text_area(
            "Job Description",
            value=default_jd,
            height=200,
            help="Paste the full job description here.",
        )

        st.markdown("---")
        st.subheader("Generative AI")

        # Load API key securely from secrets, fall back to user input
        try:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        except FileNotFoundError:
            gemini_api_key = ""

        if gemini_api_key:
            st.success("API Key successfully loaded from secure storage.")
            configure_gemini(gemini_api_key)
        else:
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Required for advanced GenAI candidate analysis, interview prep, and email generation.",
            )
            if gemini_api_key:
                configure_gemini(gemini_api_key)

        st.markdown("---")
        st.subheader("Model Evaluation")
        if st.button("Run evaluation on training dataset"):
            try:
                metrics = evaluate_model()
                st.success(
                    f"Accuracy: {metrics['accuracy']:.3f}, "
                    f"Precision: {metrics['precision']:.3f}, "
                    f"Recall: {metrics['recall']:.3f}, "
                    f"F1: {metrics['f1']:.3f}"
                )
            except Exception as e:
                st.error(f"Could not evaluate model: {e}")

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload one or more resume PDFs to get started.")
        return

    if not job_description.strip():
        st.warning("Please enter a job description in the sidebar before analyzing.")
        return

    analyze_button = st.button("Analyze Candidates", type="primary")
    if not analyze_button:
        return

    resume_names: List[str] = []
    resume_texts: List[str] = []

    with st.spinner("Extracting text from PDFs..."):
        for f in uploaded_files:
            text = extract_text_from_pdf(f)
            resume_names.append(f.name.replace(".pdf", ""))
            resume_texts.append(text)

    if not any(resume_texts):
        st.error("Could not extract text from any of the uploaded PDFs.")
        return

    clf, vectorizer = load_model_and_vectorizer()

    with st.spinner("Computing similarity scores and analyzing skills..."):
        sim_scores_raw = compute_job_resume_similarity(job_description, resume_texts)
        tfidf_scores_pct = [round(s * 100, 2) for s in sim_scores_raw]

        bert_scores_raw, bert_error = compute_bert_similarity(job_description, resume_texts)
        bert_scores_pct = [round(s * 100, 2) for s in bert_scores_raw]
        if bert_error:
            st.warning(f"BERT similarity could not be computed: {bert_error}. Using zeros.")

        skill_analyses = [
            analyze_skill_match(resume_texts[i], job_description)
            for i in range(len(resume_texts))
        ]

        summaries = [summarize_resume(t) for t in resume_texts]

        model_probs: List[float | None] = [None] * len(resume_texts)
        if clf is not None and vectorizer is not None:
            combined_texts = [
                preprocess_text(resume_texts[i]) + " " + preprocess_text(job_description)
                for i in range(len(resume_texts))
            ]
            X_vec = vectorizer.transform(combined_texts)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_vec)[:, 1]
                model_probs = [float(p) for p in probs]

        score_breakdowns: List[ScoreBreakdown] = []
        combined_final_scores_pct: List[float] = []
        for i in range(len(resume_texts)):
            breakdown = compute_final_score(
                similarity_score=sim_scores_raw[i],
                skills_score=skill_analyses[i].match_score,
                model_probability=model_probs[i],
            )
            score_breakdowns.append(breakdown)
            combined_score = 0.5 * sim_scores_raw[i] + 0.5 * bert_scores_raw[i]
            combined_final_scores_pct.append(round(combined_score * 100, 2))

        duplicate_pairs = compute_resume_to_resume_similarity(resume_texts)
        duplicates = [(i, j, s) for (i, j, s) in duplicate_pairs if s >= 0.9]

    # Build results table.
    rows = []
    for idx, name in enumerate(resume_names):
        skills_str = ", ".join(skill_sets_to_strings(skill_analyses[idx].detected_skills))
        missing_str = ", ".join(skill_sets_to_strings(skill_analyses[idx].missing_skills))
        breakdown = score_breakdowns[idx]
        explanation = explain_candidate(
            name=name,
            skills_found=skill_sets_to_strings(skill_analyses[idx].detected_skills),
            missing_skills=skill_sets_to_strings(skill_analyses[idx].missing_skills),
            tfidf_score_pct=tfidf_scores_pct[idx],
            bert_score_pct=bert_scores_pct[idx],
            final_score_pct=combined_final_scores_pct[idx],
        )
        rows.append(
            {
                "Candidate": name,
                "TF-IDF Score %": round(tfidf_scores_pct[idx], 2),
                "BERT Score %": round(bert_scores_pct[idx], 2),
                "Combined Final Score %": combined_final_scores_pct[idx],
                "Legacy Similarity %": breakdown.similarity_score,
                "Skills Match %": breakdown.skills_score,
                "Model Probability": breakdown.model_probability,
                "Legacy Final Score %": breakdown.final_score,
                "Skills Found": skills_str,
                "Missing Skills": missing_str,
                "Summary": summaries[idx],
                "Explanation": explanation,
                "Raw Text": resume_texts[idx],
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.sort_values(by="Combined Final Score %", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    top_candidate = results_df.iloc[0]

    st.success("Analysis complete.")

    st.subheader("Top Candidate")
    st.markdown(
        f"**{top_candidate['Candidate']}** — Final Combined Score: "
        f"**{top_candidate['Combined Final Score %']:.2f}%**"
    )
    st.write(top_candidate["Explanation"])

    st.subheader("Deep Learning Analysis")
    st.info(
        "BERT captures semantic meaning beyond keywords. "
        "Final combined score = 0.5 * TF-IDF score + 0.5 * BERT score."
    )

    st.subheader("Ranking Table")
    st.dataframe(
        results_df[
            [
                "Candidate",
                "TF-IDF Score %",
                "BERT Score %",
                "Combined Final Score %",
                "Skills Match %",
                "Model Probability",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Score Distribution")
    st.bar_chart(
        data=results_df.set_index("Candidate")[["Combined Final Score %"]],
        use_container_width=True,
    )

    st.subheader("Detailed Candidate Breakdown")
    for _, row in results_df.iterrows():
        with st.expander(f"{row['Candidate']} — {row['Combined Final Score %']:.2f}%"):
            st.markdown("**Resume Summary**")
            st.write(row["Summary"])

            st.markdown("**Skills Found**")
            st.write(row["Skills Found"] or "None detected.")

            st.markdown("**Missing Skills (Gap Analysis)**")
            st.write(row["Missing Skills"] or "No obvious gaps relative to the job description.")

            st.markdown("**Score Breakdown**")
            st.write(
                f"TF-IDF Score: {row['TF-IDF Score %']:.2f}%  \n"
                f"BERT Score: {row['BERT Score %']:.2f}%  \n"
                f"Combined Final Score: {row['Combined Final Score %']:.2f}%  \n"
                f"(Legacy) Similarity: {row['Legacy Similarity %']:.2f}%  \n"
                f"Skills Match: {row['Skills Match %']:.2f}%  \n"
                f"Model Probability: {row['Model Probability'] if row['Model Probability'] is not None else 'N/A'}  \n"
                f"(Legacy) Final Score: {row['Legacy Final Score %']:.2f}%"
            )

            st.markdown("**Explanation**")
            st.write(row["Explanation"])

            st.markdown("---")
            st.markdown("### Generative AI Features")
            if not gemini_api_key:
                st.info(
                    "Enter your Gemini API Key in the sidebar to unlock GenAI Deep Analysis, "
                    "Interview Questions, and Email Drafts."
                )
            else:
                tab1, tab2, tab3 = st.tabs(["Deep Analysis", "Interview Prep", "Email Drafts"])

                with tab1:
                    analysis_key = f"analysis_{row['Candidate']}"
                    if st.button("Generate Deep Analysis", key=f"btn_{analysis_key}"):
                        with st.spinner("Analyzing candidate..."):
                            st.session_state[analysis_key] = generate_candidate_analysis(
                                row["Raw Text"], job_description
                            )
                    if analysis_key in st.session_state:
                        st.markdown(st.session_state[analysis_key])

                with tab2:
                    questions_key = f"questions_{row['Candidate']}"
                    if st.button("Generate Interview Questions", key=f"btn_{questions_key}"):
                        with st.spinner("Generating questions..."):
                            st.session_state[questions_key] = generate_interview_questions(
                                row["Raw Text"], job_description
                            )
                    if questions_key in st.session_state:
                        st.markdown(st.session_state[questions_key])

                with tab3:
                    email_accept_key = f"email_accept_{row['Candidate']}"
                    email_reject_key = f"email_reject_{row['Candidate']}"

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Draft Acceptance Email", key=f"btn_{email_accept_key}"):
                            with st.spinner("Drafting email..."):
                                st.session_state[email_accept_key] = generate_email_draft(
                                    row["Candidate"], row["Raw Text"], job_description, "accept"
                                )
                        if email_accept_key in st.session_state:
                            st.markdown(st.session_state[email_accept_key])
                    with col2:
                        if st.button("Draft Rejection Email", key=f"btn_{email_reject_key}"):
                            with st.spinner("Drafting email..."):
                                st.session_state[email_reject_key] = generate_email_draft(
                                    row["Candidate"], row["Raw Text"], job_description, "reject"
                                )
                        if email_reject_key in st.session_state:
                            st.markdown(st.session_state[email_reject_key])

    if duplicates:
        st.subheader("Potential Duplicate Resumes")
        dup_rows: List[Tuple[str, str, float]] = []
        for i, j, s in duplicates:
            dup_rows.append((resume_names[i], resume_names[j], s))
        dup_df = pd.DataFrame(dup_rows, columns=["Resume A", "Resume B", "Similarity"])
        st.dataframe(dup_df, use_container_width=True)
    else:
        st.info("No highly similar (duplicate) resumes detected.")

    st.subheader("Download Results")
    csv_bytes = dataframe_to_csv_download(results_df)
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )

    log_event(
        "analysis_finished",
        {
            "num_candidates": int(results_df.shape[0]),
            "top_candidate": str(top_candidate["Candidate"]),
        },
    )


if __name__ == "__main__":
    main()
