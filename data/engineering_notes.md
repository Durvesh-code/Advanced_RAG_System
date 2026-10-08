# Engineering Notes

The ingestion layer supports Markdown, text, and PDF documents. PDF chunks retain filename and page metadata so answers can be traced back to the source.

Dense embeddings are stored in a transparent NumPy matrix for this prototype, while BM25 is rebuilt from the stored chunk text on startup.

Configuration is driven by environment variables, allowing retrieval depth, chunk sizing, model choices, and the generation model to be changed without modifying source code.
