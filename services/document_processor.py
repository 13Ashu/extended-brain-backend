"""
Document Processing Utilities
Extract text from PDFs, Word docs, etc.

IMPORTANT: Save this file as services/document_processor.py
Create a 'services' folder and put this file inside it
"""

import httpx
from typing import Optional
import io


async def extract_document_text(url: str) -> str:
    """
    Extract text from document URL
    Supports: PDF, DOCX
    """
    
    try:
        # Download document
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content = response.content
        
        # Determine file type from URL or content
        if url.lower().endswith('.pdf'):
            return await extract_pdf_text(content)
        elif url.lower().endswith(('.docx', '.doc')):
            return await extract_docx_text(content)
        else:
            return "[Document content - unsupported format]"
    
    except Exception as e:
        print(f"Error extracting document text: {e}")
        return "[Error extracting document content]"


async def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        from PyPDF2 import PdfReader
        
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return "[Error reading PDF]"


async def extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX bytes"""
    try:
        from docx import Document
        
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)
        
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return "[Error reading Word document]"
