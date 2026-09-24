import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000/retrieve"


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ChunkBench",
    page_icon="",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .result-card {
        padding: 0.5rem 0;
    }

    .score {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .chunk-text {
        font-size: 1rem;
        line-height: 1.65;
    }

    .metadata-label {
        font-size: 0.85rem;
        color: #999;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper: display one retrieved result
# ============================================================

def display_result(result, rank):
    """
    Display one retrieved chunk.

    Expected result format:

    {
        "score": 0.77,
        "text": "...",
        "metadata": {...}
    }

    Also handles Qdrant-style payloads if necessary.
    """

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = result.get("score", 0)

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    st.markdown(
        f"**#{rank} — Score: {score:.4f}**"
    )

    # --------------------------------------------------------
    # Extract payload
    # --------------------------------------------------------

    payload = result.get("payload", {})

    if not isinstance(payload, dict):
        payload = {}

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text = result.get("text")

    if text is None:
        text = payload.get("text", "")

    if not text:
        text = "No chunk text returned."

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    metadata = result.get("metadata")

    if metadata is None:
        metadata = payload.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    # --------------------------------------------------------
    # Main chunk content
    # --------------------------------------------------------

    st.markdown(
        f'<div class="chunk-text">{text}</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Metadata dropdown
    # --------------------------------------------------------

    with st.expander("Metadata"):

        if not metadata:
            st.write("No metadata available.")

        else:

            # Important metadata first
            important_keys = [
                "source",
                "book_name",
                "page",
                "page_no",
                "page_label",
                "chunking_method",
                "chunk_size",
                "source_document",
            ]

            shown = set()

            # ----------------------------------------------
            # Important fields
            # ----------------------------------------------

            for key in important_keys:

                if key in metadata:

                    value = metadata[key]

                    st.markdown(
                        f"**{key}:** {value}"
                    )

                    shown.add(key)

            # ----------------------------------------------
            # Remaining metadata
            # ----------------------------------------------

            remaining = {
                key: value
                for key, value in metadata.items()
                if key not in shown
            }

            if remaining:

                st.markdown("---")

                st.caption("Other metadata")

                for key, value in remaining.items():

                    st.markdown(
                        f"**{key}:** {value}"
                    )

    st.divider()


# ============================================================
# Header
# ============================================================

st.title("ChunkBench")

st.write("Compare retrieval results from different chunking strategies.")


# ============================================================
# Query input
# ============================================================

query = st.text_input(
    "Enter your question",
    placeholder="Ask something about the documents...",
)


# ============================================================
# Retrieve button
# ============================================================

if st.button("Retrieve", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

        st.stop()

    # --------------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------------

    try:

        with st.spinner("Retrieving relevant chunks..."):

            response = requests.post(
                BACKEND_URL,
                json={
                    "query": query
                },
                timeout=120,
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI backend."
        )

        st.info(
            "Make sure FastAPI is running on port 8000."
        )

        st.stop()

    except requests.exceptions.Timeout:

        st.error(
            "The retrieval request timed out."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        st.error(
            f"Request failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Handle backend errors BEFORE response.json()
    # --------------------------------------------------------

    if not response.ok:

        st.error(
            f"Backend returned HTTP {response.status_code}"
        )

        # Show actual backend error
        if response.text:

            st.code(
                response.text,
                language="text",
            )

        st.stop()

    # --------------------------------------------------------
    # Parse JSON safely
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        st.error(
            "Backend returned a response that is not valid JSON."
        )

        st.code(
            response.text,
            language="text",
        )

        st.stop()


    # ========================================================
    # Results
    # ========================================================

    recursive_results = data.get(
        "recursive",
        []
    )

    semantic_results = data.get(
        "semantic",
        []
    )


    # ========================================================
    # Two-column comparison
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Recursive
    # --------------------------------------------------------

    with col1:

        st.subheader("Recursive Chunking")

        if not recursive_results:

            st.info(
                "No recursive results found."
            )

        else:

            for rank, result in enumerate(
                recursive_results,
                start=1,
            ):

                display_result(
                    result,
                    rank,
                )


    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    with col2:

        st.subheader("Semantic Chunking")

        if not semantic_results:

            st.info(
                "No semantic results found."
            )

        else:

            for rank, result in enumerate(
                semantic_results,
                start=1,
            ):

                display_result(
                    result,
                    rank,
                )