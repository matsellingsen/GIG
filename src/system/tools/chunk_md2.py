"""Chunking script for Hugo Markdown files, based on shortcodes. This is used to split the content into manageable pieces for processing by the agent."""
import re
import os
import sys

def get_foldernames(article_path):
    """extract name of all parent-folders up until the content-folder, to be used as metadata for the agent."""
    content_folder = "content"
    foldernames = []
    current_path = os.path.dirname(article_path)
    while current_path != os.path.dirname(current_path):  # While not at the root
        foldername = os.path.basename(current_path)
        if foldername == content_folder:
            break
        foldernames.append(foldername)
        current_path = os.path.dirname(current_path)
    return foldernames

def chunk_article(article_text):
    """Chunk the article text based on Hugo shortcodes. This function identifies the shortcodes and splits the content accordingly.
    """
    # Regular expression to match Hugo shortcodes
    shortcode_pattern = re.compile(r'{{<\s*(\w+)(.*?)\s*>}}(.*?){{<\s*/\s*\1\s*>}}', re.DOTALL)
    
    # Find all matches of the shortcode pattern
    matches = shortcode_pattern.findall(article_text)
    
    # Create a list to hold the chunks
    chunks = []
    
    for match in matches:
        shortcode_name = match[0]
        shortcode_args = match[1].strip()
        shortcode_content = match[2].strip()
        
        # Create a chunk for this shortcode
        chunk = {
            "shortcode_name": shortcode_name,
            "shortcode_args": shortcode_args,
            "shortcode_content": shortcode_content
        }
        chunks.append(chunk)
    
    return chunks

