import aiofiles
import urllib
import mistune
import os

async def write_to_file(filename: str, text: str) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Convert text to UTF-8, replacing any problematic characters
    text_utf8 = text.encode('utf-8', errors='replace').decode('utf-8')

    async with aiofiles.open(filename, "w", encoding='utf-8') as file:
        await file.write(text_utf8)

async def write_text_to_md(text: str, filename: str = "") -> str:
    """Writes text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    file_path = f"outputs/{filename[:60]}.md"
    await write_to_file(file_path, text)
    return urllib.parse.quote(file_path)

async def write_md_to_pdf(text: str, filename: str = "") -> str:
    """Converts Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated PDF.
    """
    file_path = f"outputs/{filename[:60]}.pdf"

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "styles", "pdf_styles.css")

        from md2pdf.core import md2pdf
        md2pdf(file_path,
               md_content=text,
               css_file_path=css_path,
               base_url=None)
        print(f"Report written to {file_path}")
        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")

    try:
        from bs4 import BeautifulSoup
        import fitz

        html = mistune.html(text)
        soup = BeautifulSoup(html, "html.parser")
        plain_text = soup.get_text("\n")

        doc = fitz.open()
        page = doc.new_page()
        font_size = 11
        line_height = font_size * 1.4
        margin_x = 72
        margin_y = 72

        def new_page():
            return doc.new_page()

        max_width = page.rect.width - (margin_x * 2)
        max_height = page.rect.height - margin_y
        y = margin_y

        for raw_line in plain_text.splitlines():
            line = raw_line.rstrip()
            if not line:
                y += line_height
                if y > max_height:
                    page = new_page()
                    y = margin_y
                continue

            remaining = line
            while remaining:
                if fitz.get_text_length(remaining, fontsize=font_size) <= max_width:
                    chunk = remaining
                    remaining = ""
                else:
                    words = remaining.split(" ")
                    current = ""
                    for word in words:
                        candidate = f"{current} {word}".strip()
                        if fitz.get_text_length(candidate, fontsize=font_size) <= max_width:
                            current = candidate
                        else:
                            break
                    if not current:
                        current = remaining[: max(1, int(len(remaining) * 0.5))]
                        while fitz.get_text_length(current, fontsize=font_size) > max_width and len(current) > 1:
                            current = current[:-1]
                    remaining = remaining[len(current):].lstrip()
                    chunk = current

                if y > max_height:
                    page = new_page()
                    y = margin_y
                page.insert_text((margin_x, y), chunk, fontsize=font_size)
                y += line_height

        doc.save(file_path)
        print(f"Report written to {file_path}")
        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path
    except Exception as e:
        print(f"Error in fallback PDF generation: {e}")
        return ""

async def write_md_to_word(text: str, filename: str = "") -> str:
    """Converts Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated DOCX.
    """
    file_path = f"outputs/{filename[:60]}.docx"

    try:
        from docx import Document
        from htmldocx import HtmlToDocx
        # Convert report markdown to HTML
        html = mistune.html(text)
        # Create a document object
        doc = Document()
        # Convert the html generated from the report to document format
        HtmlToDocx().add_html_to_document(html, doc)

        # Saving the docx document to file_path
        doc.save(file_path)

        print(f"Report written to {file_path}")

        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path

    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e}")
        return ""
