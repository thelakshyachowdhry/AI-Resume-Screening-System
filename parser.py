import io
from typing import Union

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_obj: Union[io.BytesIO, str]) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    file_obj : Union[io.BytesIO, str]
        Either a file-like object (as provided by Streamlit upload)
        or a file path string on disk.

    Returns
    -------
    str
        Extracted text concatenated across all pages. May be an empty
        string if no extractable text is found.
    """
    try:
        if isinstance(file_obj, (io.BytesIO, io.BufferedReader)):
            reader = PdfReader(file_obj)
        else:
            reader = PdfReader(str(file_obj))
    except Exception:
        return ""

    all_text: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            all_text.append(page_text)

    return "\n".join(all_text).strip()

