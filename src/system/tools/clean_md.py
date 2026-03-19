import re
import html
from bs4 import BeautifulSoup

def clean_markdown_chunk(text: str) -> str:
    """
    Cleans a Markdown/Hugo chunk for LLM processing:
    1. Removes HTML comments.
    2. Converts meaningful HTML (h1-h6) to Markdown.
    3. Strips layout tags (div, span) but keeps content.
    4. Preserves Hugo shortcode structure but reduces syntax noise if needed.
    """
    
    # 1. Remove HTML comments <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # 2. Convert HTML headers and layout tags. 
    try:
        # Pre-process: Hugo shortcodes {{< ... >}} are not valid HTML and get escaped by BS4.
        # We replace them with a temporary placeholder that BS4 won't mangle.
        # However, since they can be multi-line or contain nested content, simple replacement is risky.
        # A safer bet is to let BS4 run, then unescape the specific escaped sequences for shortcodes.
        
        soup = BeautifulSoup(text, 'html.parser')

        # Convert HTML headers to Markdown
        for i in range(1, 7):
            for tag in soup.find_all(f'h{i}'):
                md_header = '#' * i + ' ' + tag.get_text().strip() + '\n'
                tag.replace_with(md_header)

        # formatting: convert <b>/<strong> to **text**
        for tag in soup.find_all(['b', 'strong']):
            tag.replace_with(f"**{tag.get_text()}**")
        
        # formatting: convert <i>/<em> to *text*
        for tag in soup.find_all(['i', 'em']):
            tag.replace_with(f"*{tag.get_text()}*")
            
        # formatting: convert <sup> to ^(text) or just text
        for tag in soup.find_all(['sup']):
            tag.replace_with(f"^({tag.get_text()})")

        # Unwrap layout tags (div, span, section, article) - keep children
        for tag in soup.find_all(['div', 'span', 'section', 'article']):
            tag.unwrap()

        # Get the cleaned text
        clean_text = str(soup)
        
        # Post-process: Unescape the Hugo shortcode brackets that BS4 escaped
        # {{&lt; -> {{<  and  &gt;}} -> >}}
        clean_text = clean_text.replace("{{&lt;", "{{<").replace("&gt;}}", ">}}")
        # Also fix the closing shortcode pattern if it was escaped differently (e.g. {{&lt; /textimage &gt;}})
        clean_text = re.sub(r"{{&lt;\s*/", "{{< /", clean_text)
        clean_text = re.sub(r"&gt;}}", ">}}", clean_text) # redundant but safe

        # General unescape for other common entities might be too aggressive, 
        # so we stick to fixing what BS4 broke regarding shortcodes.

    except ImportError:
        # Fallback if bs4 is missing
        clean_text = text
        # Simple regex stripping for common tags
        clean_text = re.sub(r'<(div|span|section|article)[^>]*>', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'</(div|span|section|article)>', '', clean_text, flags=re.IGNORECASE)

    # 3. Collapse multiple newlines (common after stripping tags)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    return clean_text.strip()
