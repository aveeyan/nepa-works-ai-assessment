# src/nw_rag/frontend/app.py

# Standard Imports
import os
import sys
import json
import requests
import webbrowser
from typing import Tuple, Dict, Any, Generator
from pathlib import Path

# Third-Party Imports
import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local Imports
from nw_rag.logging_config import StructuredLogger, logger


class RAGInterface:
    """Gradio interface for the RAG system."""

    def __init__(self):
        self.api_url = os.getenv("API_URL", "http://localhost:8000/api/v1/query")
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.docs_dir = self.project_root / "docs"
        self.local_mode = False

        # Check if API is available
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                logger.info("Connected to RAG API")
            else:
                self._init_local_mode()
        except:
            self._init_local_mode()

    def _init_local_mode(self):
        """Initialize local mode if API is not available."""
        logger.info("API not available, running in local mode")
        self.local_mode = True
        from nw_rag.core.ingestion import DocumentIngester
        from nw_rag.services.rag_engine import RAGEngine

        self.ingester = DocumentIngester()
        if not self.ingester.load_index():
            logger.info("Building index locally...")
            count = self.ingester.ingest()
            self.ingester.save_index()
            logger.info(f"Ingested {count} chunks locally")
        self.rag_engine = RAGEngine(self.ingester)

    def _get_file_path(self, source: str) -> str:
        """Get absolute file path from source string."""
        if source.startswith('./'):
            source = source[2:]

        file_name = Path(source).name

        # Check for PDF first
        pdf_path = self.docs_dir / f"{Path(file_name).stem}.pdf"
        if pdf_path.exists():
            return str(pdf_path.absolute())

        txt_path = self.docs_dir / f"{Path(file_name).stem}.txt"
        if txt_path.exists():
            return str(txt_path.absolute())

        md_path = self.docs_dir / f"{Path(file_name).stem}.md"
        if md_path.exists():
            return str(md_path.absolute())

        original_path = self.docs_dir / source
        if original_path.exists():
            return str(original_path.absolute())

        return None

    def _open_file(self, file_path: str):
        """Open file using system default application."""
        if file_path and os.path.exists(file_path):
            webbrowser.open(f"file://{file_path}")
        return gr.update()

    def _format_result_with_links(self, result: Dict[str, Any]) -> str:
        """Format the response with clickable source links."""
        answer = result.get('answer', 'No answer')

        if result.get('fallback', False):
            answer = f"⚠️ {answer}"
            if result.get('confidence', 0) > 0:
                answer += f"\n\n**Confidence:** {result['confidence']:.3f}"

        # Add sources as HTML buttons
        if result.get('sources'):
            answer += "\n\n---\n\n### 📚 Sources\n\n"
            for i, source in enumerate(result['sources'], 1):
                source_name = source.get('source', 'unknown')
                score = source.get('score', 0)
                display_name = Path(source_name).name if source_name else 'unknown'

                file_path = self._get_file_path(source_name)
                if file_path:
                    icon = "📄" if file_path.endswith('.pdf') else "📝"
                    # Use HTML with onclick to open file
                    answer += f'{i}. **{icon} <a href="#" onclick="window.open(\'{file_path}\', \'_blank\'); return false;">{display_name}</a>** (score: {score:.3f})\n'
                else:
                    answer += f"{i}. **📄 {display_name}** (score: {score:.3f})\n"

        # Add metrics
        timing = result.get('timing', {})
        answer += f"\n\n### ⏱️ Metrics\n"
        answer += f"- Retrieval: {timing.get('retrieval', 0):.3f}s\n"
        answer += f"- Generation: {timing.get('generation', 0):.3f}s\n"
        answer += f"- Total: {timing.get('total', 0):.3f}s\n"

        tokens = result.get('tokens', {})
        answer += f"- Tokens: {tokens.get('prompt', 0) + tokens.get('completion', 0)}"

        return answer

    def query_api_stream(self, question: str) -> Generator[str, None, None]:
        """Query the API with streaming."""
        try:
            response = requests.post(
                self.api_url,
                json={"question": question, "stream": True},
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            full_answer = ""
            metadata = None

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'chunk' in data:
                                full_answer += data['chunk']
                                yield full_answer
                            elif 'done' in data:
                                metadata = data.get('metadata', {})
                        except json.JSONDecodeError:
                            continue

            if metadata:
                result = {
                    "answer": full_answer,
                    "sources": metadata.get('sources', []),
                    "confidence": metadata.get('confidence', 0),
                    "timing": metadata.get('timing', {}),
                    "tokens": metadata.get('tokens', {"prompt": 0, "completion": 0}),
                    "fallback": metadata.get('fallback', False)
                }
                yield self._format_result_with_links(result)

        except requests.exceptions.RequestException as e:
            logger.error(f"API error: {e}")
            yield f"❌ Error connecting to API: {str(e)}"

    def query_api_nonstream(self, question: str) -> str:
        """Query the API without streaming."""
        try:
            response = requests.post(
                self.api_url,
                json={"question": question, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return self._format_result_with_links(result)

        except requests.exceptions.RequestException as e:
            logger.error(f"API error: {e}")
            return f"❌ Error connecting to API: {str(e)}"

    def query_local(self, question: str, stream: bool = False) -> str:
        """Query locally."""
        result = self.rag_engine.query(question)
        return self._format_result_with_links(result)

    def process_query(self, question: str, stream: bool) -> str:
        """Process a query and return formatted response."""
        if not question or question.strip() == "":
            return "Please enter a question."

        StructuredLogger.log_request(question, stream)

        try:
            if self.local_mode:
                return self.query_local(question, stream)
            else:
                if stream:
                    full_response = ""
                    for chunk in self.query_api_stream(question):
                        full_response = chunk
                    return full_response
                else:
                    return self.query_api_nonstream(question)

        except Exception as e:
            logger.error(f"Query error: {e}")
            StructuredLogger.log_error(e, {"question": question})
            return f"❌ Error: {str(e)}"


# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface."""

    rag_interface = RAGInterface()

    # Custom CSS for source links
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header-text {
        text-align: center !important;
        font-size: 28px !important;
        font-weight: bold !important;
        background: linear-gradient(135deg, #f5af19, #f12711);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .subheader-text {
        text-align: center !important;
        color: #666 !important;
        margin-bottom: 20px !important;
    }
    footer {
        text-align: center !important;
        padding: 20px !important;
        color: #999 !important;
        font-size: 12px !important;
    }
    a {
        color: #f5af19 !important;
        text-decoration: none !important;
        cursor: pointer !important;
    }
    a:hover {
        text-decoration: underline !important;
        color: #f12711 !important;
    }
    """

    with gr.Blocks(
        title="Aperture Science Document Intelligence",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        # Header
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px 0;">
                <h1 class="header-text">🔬 APERTURE SCIENCE</h1>
                <h2 class="header-text" style="font-size: 22px;">DOCUMENT INTELLIGENCE</h2>
                <p class="subheader-text">Ask questions about Aperture Science technologies and documentation</p>
                <p style="color: #888; font-size: 14px;">📄 Click source links to open PDF documents</p>
            </div>
            """
        )

        # Main layout
        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What would you like to know about Aperture Science?",
                    lines=3,
                    max_lines=5
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask", variant="primary", scale=2)
                    stream_checkbox = gr.Checkbox(label="Stream Response", value=False, scale=1)

                # Example questions
                gr.Markdown("### 💡 Example Questions")
                example_buttons = gr.Row()

                examples = [
                    "What is the portal gun?",
                    "How do I handle a weighted storage cube?",
                    "What are the emergency procedures?",
                    "Tell me about GLaDOS",
                    "What is the cake recipe?",
                    "What are the safety protocols?"
                ]

                example_btn_list = []
                for example in examples:
                    btn = gr.Button(example, size="sm", variant="secondary")
                    example_btn_list.append(btn)

                def set_example(example):
                    return gr.update(value=example)

                for btn, example in zip(example_btn_list, examples):
                    btn.click(
                        fn=lambda q=example: q,
                        outputs=question_input
                    )

            with gr.Column(scale=6):
                answer_output = gr.Markdown(
                    label="Answer",
                    value="Ask a question to get started...\n\n*📄 Click source links to open documents*",
                    show_label=True
                )

        # Footer
        gr.Markdown(
            """
            <footer>
                <p>Powered by Llama 3.2, FAISS, and Aperture Science Technology</p>
                <p>📄 Click source links to open the original documents</p>
                <p>⚠️ The cake is a lie. Or is it?</p>
            </footer>
            """
        )

        # Handle submission
        def handle_submit(question, stream):
            return rag_interface.process_query(question, stream)

        submit_btn.click(
            fn=handle_submit,
            inputs=[question_input, stream_checkbox],
            outputs=answer_output
        )

        question_input.submit(
            fn=handle_submit,
            inputs=[question_input, stream_checkbox],
            outputs=answer_output
        )

        # Also handle example button clicks to automatically submit
        for btn in example_btn_list:
            btn.click(
                fn=lambda: gr.update(),
                inputs=[],
                outputs=[question_input]
            ).then(
                fn=handle_submit,
                inputs=[question_input, stream_checkbox],
                outputs=answer_output
            )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
