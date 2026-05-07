import sys
import builtins
import os
# ensure project root is on sys.path so we can import modules
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base)
from pipelines.inference_module import InferenceModule

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: run_pipeline_with_question.py "<atomic question>"')
        sys.exit(1)
    q = sys.argv[1]

    # Monkeypatch input
    orig_input = builtins.input
    builtins.input = lambda prompt='': q
    try:
        im = InferenceModule()
        im.run()
    finally:
        builtins.input = orig_input
