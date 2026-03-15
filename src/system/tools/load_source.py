"""Get (part of) the unstructured natural language text to 
be used as the source of the OWL 2 KB."""

"""This is a placeholder implementation that simply returns a hardcoded string. 
In a real implementation, this function should read from a file, 
database, or other source."""
def load_source(file_path: str = None) -> str:
    if file_path:
        with open(file_path, "r") as f:
            dev_text = f.read()
    else:
        dev_text = """SENS Motion is an integrated system for collecting physical 
        activity data from groups of people."""
    return dev_text


