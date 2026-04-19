import os

def resolve_ttl_path(ttl_path: str = None) -> str:
    """
    Resolves the TTL file path, handling both defined and None.
    Args:
        ttl_path (str): The provided TTL file path. If None, the function will attempt to find a default TTL file in a predefined directory."""
    
    default_ttl_path = "C:\\Users\\matse\\gig\\src\\system_v5\\KB\\current"
    resolved_ttl_path = None # Initialize resolved_ttl_path to None

    if ttl_path is None:
        ttl_file_name = [f for f in os.listdir(default_ttl_path) if f.endswith(".ttl")]
        if ttl_file_name:
            resolved_ttl_path = os.path.join(default_ttl_path, ttl_file_name[0])
        if resolved_ttl_path is None:
            raise ValueError("No TTL path provided and no default TTL file found in the directory.")
        return resolved_ttl_path

    # if path is provided, simply return it (assuming the caller will handle any path issues)
    resolved_ttl_path = ttl_path
    return resolved_ttl_path
    
    
