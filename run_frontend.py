# run_frontend.py

# Standard Imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Third-Party Imports
import gradio as gr

# Local Imports
from nw_rag.frontend.app import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
