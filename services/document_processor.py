"""
Document Processing Utilities
Extract text from PDFs and Word docs.

Two entry points:
  - extract_text_from_bytes(content, hint)  ← preferred; works on raw bytes already
    in hand (e.g. loaded from the StoredImage table on the iOS capture path).
  - extract_document_text(url)              ← legacy; downloads an EXTERNAL url first
    (Telegram/WhatsApp media). Internal /api/images/{id} urls are auth-gated, so the
    capture path should use the bytes-based function instead of this one.

File type is detected by magic bytes (reliable) and falls back to a filename/url
hint, because our internal media urls (/api/images/{id}) carry no file extension.
"""

import httpx
import io


def detect_doc_kind(content: bytes, hint: str = "") -> str:
    """Return 'pdf' | 'docx' | 'unknown' from magic bytes, with a filename/url fallback."""
    head = content[:8] if content else b""
    # PDF files start with "%PDF"
    if head.startswith(b"%PDF"):
        return "pdf"
    # DOCX (and any OOXML) is a ZIP archive: "PK\x03\x04"
    if head.startswith(b"PK\x03\x04"):
        # Could be docx/xlsx/pptx — assume docx for our text use case
        return "docx"
    # Fall back to the hint (filename or url)
    h = (hint or "").lower()
    if h.endswith(".pdf"):
        return "pdf"
    if h.endswith((".docx", ".doc")):
        return "docx"
    return "unknown"


async def extract_text_from_bytes(content: bytes, hint: str = "") -> str:
    """Extract text from raw document bytes already in hand."""
    if not content:
        return "[Empty document]"
    kind = detect_doc_kind(content, hint)
    if kind == "pdf":
        return await extract_pdf_text(content)
    if kind == "docx":
        return await extract_docx_text(content)
    return "[Document content - unsupported format]"


async def extract_document_text(url: str) -> str:
    """Download an external document URL and extract its text. Legacy webhook path."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content = response.content
        return await extract_text_from_bytes(content, hint=url)
    except Exception as e:
        print(f"Error extracting document text: {e}")
        return "[Error extracting document content]"


async def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        try:
            from pypdf import PdfReader          # modern package name
        except ImportError:
            from PyPDF2 import PdfReader          # fallback for older installs

        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return "[Error reading PDF]"


async def extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        from docx import Document

        doc = Document(io.BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return "[Error reading Word document]"
