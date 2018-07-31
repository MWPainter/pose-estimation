import sys
import os

# Add paths to system path, so can use relative imports (inside the previous repos)
sys.path.append(os.path.join(os.path.dirname(__file__), "twod_threed"))
sys.path.append(os.path.join(os.path.dirname(__file__), "stacked_hourglass"))

# Add our path name to be able to use relative imports overall
sys.path.append(os.path.dirname(__file__))
