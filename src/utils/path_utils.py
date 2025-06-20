import os
import re

def extract_dive_prefix(path: str) -> str:
    """
    Extract dive number prefix from a path (e.g., 'DIVE012')
    
    Args:
        path: Path that might contain a dive number
        
    Returns:
        Dive number string or empty string if not found
    """
    if not path:
        return ""
    
    # Normalize path separators
    normalized_path = path.replace('\\', '/')
    
    # Try to find DIVE pattern followed by numbers - case insensitive
    dive_match = re.search(r'(DIVE\d+)', normalized_path.upper())
    if dive_match:
        # Return the matched pattern in uppercase
        return dive_match.group(1)
    
    # Also try lowercase and capitalized versions
    dive_match = re.search(r'(dive\d+|Dive\d+)', normalized_path)
    if dive_match:
        # Convert the matched pattern to uppercase
        return dive_match.group(1).upper()
    
    return ""
    
