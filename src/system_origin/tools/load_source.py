"""Get (part of) the unstructured natural language text to 
be used as the source of the OWL 2 KB."""

"""This is a placeholder implementation that simply returns a hardcoded string. 
In a real implementation, this function should read from a file, 
database, or other source."""
def load_source(file_path: str = None) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            dev_text = f.read()
    else:
        dev_text = """SENS Motion is an integrated system for collecting physical activity data from groups of people. 
        It consists of a wireless activity sensor that automatically transfers data to a secure cloud. 
        It is especially well suited for use in the healthcare sector and for large research projects."""
    return dev_text



