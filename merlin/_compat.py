import sys
import importlib

# Remove pip-installed cloelib if already loaded
if "cloelib" in sys.modules:
    del sys.modules["cloelib"]

# Force Python to search the local project first
if "/home/awernersson/projects" not in sys.path:
    sys.path.insert(0, "/home/awernersson/projects")

# Import local cloelib to confirm it's loaded from the right place
import cloelib
