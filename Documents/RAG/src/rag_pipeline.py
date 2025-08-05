from parser import PDFParser
from embedder import TextEmbedder
from retriever import FaissRetriever
from generator import AnswerGenerator
import os
import numpy as np

class RAGPipeline:
    def __init__(self, data_directory: str, index_path: str = "faiss_index.bin", chunks_path: str = "text_chunks.pkl"):
        self.parser = PDFParser()
        self.embedder = TextEmbedder()
        self.retriever = FaissRetriever(index_path, chunks_path)
        self.generator = AnswerGenerator()
        self.data_directory = data_directory

    def setup_and_index(self):
        print("Starting document parsing...")
        texts = self.parser.process_all_pdfs_in_folder(self.data_directory)
        if not texts:
            print(f"No text extracted from PDF files in '{self.data_directory}'. Halting process.")
            print("Please ensure your PDF files are in the correct directory.")
            return False

        print("Starting text chunking...")
        chunks = self.embedder.chunk_multiple_texts(texts)
        if not chunks:
            print("No chunks created. Halting process.")
            return False

        print("Starting chunk embedding...")
        embeddings = self.embedder.embed_chunks(chunks)
        if not embeddings.any():
            print("Embedding creation failed. Halting process.")
            return False

        print("Building and saving FAISS index...")
        self.retriever.build_index(embeddings, chunks)
        self.retriever.save_index()
        print("Setup complete. Index is built and saved.")
        return True

    def execute_query(self, query: str) -> str:
        print(f"\nExecuting query: '{query}'")
        
        load_success = self.retriever.load_index()
        if not load_success:
            return "Failed to load the index. Please run the setup process first."

        print("Embedding the query...")
        query_embedding = self.embedder.embed_chunks([query])
        if not query_embedding.any():
            return "Failed to embed the query."
        
        print("Searching for relevant chunks...")
        retrieved_chunks_with_scores = self.retriever.search(query_embedding, k=5)
        
        if not retrieved_chunks_with_scores:
            return "Could not find relevant information."

        retrieved_chunks = [chunk for chunk, score in retrieved_chunks_with_scores]
        
        print("Generating an answer...")
        answer = self.generator.generate_answer_from_chunks(query, retrieved_chunks)
        return answer

if __name__ == '__main__':
    DATA_DIR = 'data/sample_reports'
    INDEX_FILE = "pipeline_faiss.bin"
    CHUNKS_FILE = "pipeline_chunks.pkl"

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

    print(f"--- IMPORTANT ---")
    print(f"Please make sure your financial report PDF (e.g., 'TSLA-Q1-2024-Update.pdf')")
    print(f"is placed inside the '{DATA_DIR}' directory before running.")
    print(f"-----------------")

    pipeline = RAGPipeline(data_directory=DATA_DIR, index_path=INDEX_FILE, chunks_path=CHUNKS_FILE)

    print("\n--- Step 1: Running the setup and indexing process ---")
    setup_success = pipeline.setup_and_index()

    if setup_success:
        print("\n--- Step 2: Executing sample queries on the Tesla Q1 2024 Report ---")
        
        # Query 1
        query_1 = "What was the total revenue in Q1 2024?"
        answer_1 = pipeline.execute_query(query_1)
        print("\n--- Final Answer 1 ---")
        print(answer_1)
        
        # Query 2
        query_2 = "What was the GAAP operating income in Q1?"
        answer_2 = pipeline.execute_query(query_2)
        print("\n--- Final Answer 2 ---")
        print(answer_2)

        # Query 3
        query_3 = "What was the record energy storage deployment in Q1?"
        answer_3 = pipeline.execute_query(query_3)
        print("\n--- Final Answer 3 ---")
        print(answer_3)

    else:
        print("\nHalting execution because the setup process failed.")


    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)

    print("\nPipeline testing complete.")