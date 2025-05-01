import os
from typing import List


def find_files_by_extensions(
        directory: str,
        extensions: List[str],
        recursive: bool = True
) -> List[str]:
    """
    Finds all files in the given directory that match the specified extensions.

    Args:
        directory (str): The root directory to search.
        extensions (List[str]): A list of file extensions (e.g., ['.jpg', '.png']).
        recursive (bool): Whether to search subdirectories recursively.

    Returns:
        List[str]: A list of full file paths that match the given extensions.
    """
    matches = []
    extensions = set(ext.lower() for ext in extensions)

    if not os.path.exists(directory):
        raise NotADirectoryError(f"Directory {directory} does not exist")

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    matches.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            full_path = os.path.join(directory, file)
            if os.path.isfile(full_path) and os.path.splitext(file)[1].lower() in extensions:
                matches.append(full_path)

    return matches