# passed in vars is a directory of type dir, and a pattern of type string:  
def list_files(directory: str, pattern: str) -> list[str]:
    """Scans the designated target directory for matching codebase files."""

    IGNORED_DIRS = EXCLUDE_DIRS

    print("in list_files")
    print(f"directory passed in: {directory}\n")
    print(f"pattern passed in: {pattern}\n")
    # print(f"Starting file scan in root directory: {CODE_TO_ANALYZE}")
    print(f"Starting file scan in root directory: {directory}\n")
    
    found_files = []
    base_path = Path(directory).resolve()
    print(f"base path 1: {base_path}\n")

    # Clean up the pattern in case the LLM wrapped it in extra wildcard stars
    # e.g., if pattern is "**/*.py", turn it into "*.py" to work safely with rglob
    clean_pattern = pattern.split("/")[-1] 

    found_files = []
    # Force clean recursive lookup
    for path in base_path.rglob(clean_pattern):
        if path.is_file() and not any(part in IGNORED_DIRS for part in path.parts):
            found_files.append(path)
            
    # return found_files
                
    print(f"Found {len(found_files)} relevant files in the codebase.\n")  

    return found_files

