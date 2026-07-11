# Package initialization
import sys
import os

# Add this directory to sys.path so tdb can be imported as a top-level module
_current_dir = os.path.dirname(__file__)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)