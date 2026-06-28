"""
Answer generator: takes retrieved chunks + question → grounded answer via Gemini.
"""

import os
import logging
from pathlib import Path

import google.generativeai as genai


os.environ.setdefault("GEMINI_API_KEY", "<your_api_key>")
os.environ.setdefault("GEMINI_MODEL",   "gemini-2.0-flash")

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. You answer employee questions
strictly based on the provided context excerpts. Follow these rules:

1. Only use information present in the CONTEXT sections below.
2. If the answer is not in the context, say: "I could not find this information in the available documents."
3. Be concise and factual.
4. Always cite the source document(s) and page number(s) in your answer.
5. Never hallucinate, guess, or invent information.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        context_block = "No relevant documents were found."
    else:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Context {i}]\n"
                f"Source: {c['source']} | Page: {c['page']} | Relevance: {c['score']:.2f}\n"
                f"{c['text']}"
            )
        context_block = "\n\n".join(parts)
    return (
        f"Use the following context excerpts to answer the question.\n\n"
        f"{context_block}\n\n---\n"
        f"Question: {question}\n\nAnswer (cite the source documents):"
    )


class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=  <your_api_key>"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("AnswerGenerator ready — model: %s", GEMINI_MODEL)

    def generate(
        self,
        question: str,
        chunks: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> dict:
        user_content = build_prompt(question, chunks)
        history: list[dict] = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "model" if msg["role"] == "assistant" else "user"
                history.append({"role": role, "parts": [msg["content"]]})

        answer:    str | None = None
        error_msg: str | None = None

        try:
            chat     = self.model.start_chat(history=history)
            response = chat.send_message(
                user_content,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=MAX_TOKENS, temperature=0.2,
                ),
            )
            answer = response.text.strip()

        except genai.types.BlockedPromptException as exc:
            error_msg = "Blocked by safety filters. Please rephrase."
            logger.warning("Prompt blocked: %s", exc)

        except genai.types.StopCandidateException as exc:
            error_msg = "Generation stopped unexpectedly. Please try again."
            logger.warning("Generation stopped: %s", exc)

        except Exception as exc:
            class_name = type(exc).__name__
            err_str    = str(exc)
            if "API_KEY" in err_str.upper() or "401" in err_str or "403" in err_str:
                error_msg = "Authentication failed. Check your GEMINI_API_KEY in .env."
            elif "429" in err_str or "quota" in err_str.lower():
                error_msg = "Rate limit exceeded. Wait a moment and try again."
            elif "not found" in err_str.lower() or "404" in err_str:
                error_msg = f"Model '{GEMINI_MODEL}' not found. Try gemini-1.5-flash in .env."
            else:
                error_msg = f"Gemini error ({class_name}): {err_str}"
            logger.error("Gemini call failed [%s]: %s", class_name, err_str, exc_info=True)

        if answer is None:
            answer = error_msg or "An unexpected error occurred. Please try again."

        seen: set = set()
        sources: list[dict] = []
        for c in chunks:
            key = (c["source"], c["page"])
            if key not in seen and c["score"] > 0.30:
                seen.add(key)
                sources.append({"document": c["source"], "page": c["page"]})

        confidence = 0.0
        if chunks:
            top_scores = [c["score"] for c in chunks[:3]]
            confidence = round(min(sum(top_scores) / len(top_scores), 0.99), 2)

        return {"answer": answer, "sources": sources, "confidence": confidence, "error": error_msg}