# Oracle analysis: best single and best ensemble chosen on test, same code as analysis_val/.

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "analysis_val")))

import run_all

if __name__ == "__main__":
    run_all.main(default_selection="test", out_dir=os.path.join(_HERE, "output_artifacts"))
