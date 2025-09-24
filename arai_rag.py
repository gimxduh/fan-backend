# arai_rag.py
import chromadb
chromadb.config.telemetry = False
import os
from openai import OpenAI
import re

import chromadb.segment.impl.metadata.sqlite as sqlite_module
from chromadb.utils import embedding_functions

# ----------- PATCH SQLITE DECODE -----------
def safe_decode_seq_id(seq_id_bytes):
    if isinstance(seq_id_bytes, int):
        return seq_id_bytes
    if isinstance(seq_id_bytes, (bytes, bytearray)):
        if len(seq_id_bytes) in (8, 24):
            return int.from_bytes(seq_id_bytes, "big")
    raise ValueError(f"Unexpected seq_id_bytes: {seq_id_bytes}")

sqlite_module._decode_seq_id = safe_decode_seq_id
# -------------------------------------------

# ---------------- CONFIG ----------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "fan_manual"
TOP_K = 5
# ----------------------------------------

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Init embeddings & DB
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# ---------------- HELPERS ----------------
def retrieve(query, top_k=TOP_K):
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = []
    for idx in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][idx],
            "meta": res["metadatas"][0][idx],
            "score": res["distances"][0][idx]
        })
    return docs

def split_sentences(text):
    sents = re.split(r'(?<=[\.\!\?])\s+|(?=Step \d+:)', text.strip())
    return [s.strip() for s in sents if s.strip()]

# ---------------- MAIN ANSWER ----------------
def answer_question(query, style="bullet", top_k=TOP_K):
    hits = retrieve(query, top_k=top_k)
    hits = sorted(hits, key=lambda h: h["score"])

    # 🔎 Debug log
    print("\n[DEBUG] Query:", query)
    for h in hits:
        title = (h.get("meta") or {}).get("title", "Unknown")
        print(f" - {title} | score={h['score']:.3f}")

    # dedup
    seen_texts = set()
    unique_hits = []
    for h in hits:
        text = h["text"].strip()
        if text not in seen_texts:
            unique_hits.append(h)
            seen_texts.add(text)
    hits = unique_hits

    # relax threshold → 1.5
    if not hits or hits[0]["score"] > 1.5:
        return "Sorry, I cannot answer that because it’s not in the FAN manual.", []

    # keyword guardrail for refund/return
    target_section = None
    for h in hits:
        meta = h.get("meta") or {}
        title = meta.get("title", "").lower()
        if any(kw in query.lower() for kw in ["refund", "return"]) and "refund" in title:
            target_section = h
            break
    if not target_section:
        target_section = hits[0]

    extracted = []
    if target_section:
        sents = split_sentences(target_section["text"])
        title = (target_section.get("meta") or {}).get("title", "").strip()
        sents = [s for s in sents if s and s != title]
        extracted.extend((0, s) for s in sents)

    if not extracted and hits:
        extracted = [(0, split_sentences(hits[0]["text"])[0])]

    pieces, seen = [], set()
    for hit_idx, s in extracted:
        if s not in seen:
            pieces.append((hit_idx, s))
            seen.add(s)

    title = (target_section.get("meta") or {}).get("title", "").strip()
    if style == "bullet":
        style_instr = (
            "Write the answer as Markdown bullet points. "
            "Each item should be on a new line. Do not invent extra steps."
        )
        context_text = "\n".join(f"• {s}" for _, s in pieces if s and s != title)
    else:
        style_instr = "Write the answer in 2-3 short paragraphs. Keep exact steps."
        context_text = " ".join(s for _, s in pieces if s and s != title)

    prompt = f"""You are an accurate assistant for employees.
ONLY use the excerpts below to answer the question.
Do NOT invent extra steps.

Question: {query}

Excerpts:
{context_text}

Answer format: {style_instr}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an accurate assistant for employees. ONLY use provided excerpts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0
        )
        out = response.choices[0].message.content.strip()
    except Exception as e:
        out = f"⚠️ OpenAI API call failed: {e}"

    if style == "bullet":
        out = re.sub(r"(Step \d+:)", r"\n• \1", out)
        out = re.sub(r"(•)", r"\n•", out)
        lines = [line.strip() for line in out.split("\n") if line.strip()]
        lines = [line for line in lines if title.lower() not in line.lower()]
        section_header_pattern = re.compile(r"^\d+(\.\d+)*\s*[\.:]")
        filtered = []
        for x in lines:
            if section_header_pattern.match(x):
                break
            if x.startswith("•") or x.startswith("-") or re.match(r"Step \d+:", x):
                filtered.append(x)
        for i, line in enumerate(filtered):
            if not line.startswith(("•", "-")):
                filtered[i] = "• " + line
        seen, deduped = set(), []
        for x in filtered:
            norm = re.sub(r'\s+', ' ', x[:30].lower()).strip()
            if norm not in seen and len(x) > 10:
                deduped.append(x)
                seen.add(norm)
        out = "\n".join(deduped)
    elif style == "sentence":
        out_lines = [line for line in out.split("\n") if title.lower() not in line.lower()]
        out = " ".join(out_lines).strip()

    source_ids = [{
        "section": (target_section.get("meta") or {}).get("title", "Unknown Section"),
        "preview": target_section["text"][:200]
    }]

    if not out.strip():
        section_text = target_section["text"]
        if section_text.startswith(title):
            section_text = section_text[len(title):].strip()
        out = section_text.strip()

    return out.strip(), source_ids

# ---------------- INTERACTIVE ----------------
if __name__ == "__main__":
    q = input("Enter your question: ")
    style_choice = input("Answer style? (bullet/sentence): ").strip().lower()
    ans, sources = answer_question(q, style=style_choice)
    print("\nANSWER:\n", ans)
    print("\nSOURCES:")
    for s in sources:
        print(f"- {s['section']} → {s['preview'][:80]}...\n")
