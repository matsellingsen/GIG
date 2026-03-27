"""Load preprocessed chunks from a JSONL file. This is used to load the chunks that have been preprocessed and saved by the chunking script, so that they can be used as input for the agent."""

def load_chunks(jsonl_path: str) -> list:
    """Load chunks from a JSONL file. Each line in the file should be a JSON object representing a chunk."""
    import json
    flagged_chunks = [] # chunks we want to remove
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_id"] not in flagged_chunks:
                chunks.append(chunk)
    return chunks