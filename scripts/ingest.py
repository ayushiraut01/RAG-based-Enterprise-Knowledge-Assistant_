"""
CLI script to ingest documents into the knowledge base.

Usage:
    python scripts/ingest.py --dir ./data/sample_docs
    python scripts/ingest.py --file ./data/sample_docs/HR_Policy.pdf
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory of documents to ingest")
    group.add_argument("--file", help="Single file to ingest")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.dir:
        print(f"Ingesting all documents in: {args.dir}")
        count = pipeline.ingest_directory(args.dir)
        print(f"Done. {count} new chunks added.")
    else:
        print(f"Ingesting: {args.file}")
        count = pipeline.ingest_file(args.file)
        print(f"Done. {count} new chunks added.")

    stats = pipeline.collection_stats()
    print(f"Total indexed chunks: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
