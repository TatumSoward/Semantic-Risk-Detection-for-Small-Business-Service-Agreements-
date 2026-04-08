import io
from typing import Dict, List

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="PDF Risk Analyzer", layout="wide")

RISK_HINTS = [
    "liability",
    "indemn",
    "termination",
    "breach",
    "penalty",
    "renew",
    "confidential",
    "non-compete",
    "assignment",
    "dispute",
]

KEYWORD_FALLBACK = [
    "indemnify",
    "liability",
    "penalty",
    "terminate",
    "termination",
    "auto-renew",
    "governing law",
    "confidential",
    "exclusive",
    "non-compete",
    "arbitration",
    "damages",
]


def suggest_risky_labels(all_labels: List[str]) -> List[str]:
    suggested = [
        label
        for label in all_labels
        if any(hint in label.lower() for hint in RISK_HINTS)
    ]
    if suggested:
        return suggested
    return all_labels[: min(8, len(all_labels))]


@st.cache_resource
def load_model(model_dir: str, force_cpu: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cpu"
    if not force_cpu and torch.cuda.is_available():
        device = "cuda"

    model.to(device)
    model.eval()

    id2label = model.config.id2label
    ordered_labels = [id2label[i] for i in sorted(id2label.keys())]

    return tokenizer, model, device, ordered_labels


def extract_blocks(pdf_bytes: bytes, min_chars: int) -> List[Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    segments: List[Dict] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            cleaned = " ".join(text.split())
            if len(cleaned) >= min_chars:
                segments.append(
                    {
                        "page": page_idx,
                        "bbox": [x0, y0, x1, y1],
                        "text": cleaned,
                    }
                )

    doc.close()
    return segments


def classify_segments(
    segments: List[Dict],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[Dict]:
    if not segments:
        return []

    texts = [seg["text"] for seg in segments]
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    pred_ids = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)

    id2label = model.config.id2label
    enriched = []
    for seg, pred_id, score, prob_vec in zip(segments, pred_ids, pred_scores, probs):
        item = dict(seg)
        item["pred_id"] = int(pred_id)
        item["label"] = id2label[int(pred_id)]
        item["score"] = float(score)
        item["probabilities"] = prob_vec.tolist()
        enriched.append(item)

    return enriched


def is_risky(segment: Dict, risky_labels: List[str], threshold: float, use_keywords: bool) -> bool:
    label_risky = (segment["label"] in risky_labels) and (segment["score"] >= threshold)

    if use_keywords:
        text_l = segment["text"].lower()
        keyword_risky = any(word in text_l for word in KEYWORD_FALLBACK)
        return label_risky or keyword_risky

    return label_risky


def render_highlighted_pages(pdf_bytes: bytes, risky_segments: List[Dict], zoom: float):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    risky_by_page: Dict[int, List[Dict]] = {}
    for seg in risky_segments:
        risky_by_page.setdefault(seg["page"], []).append(seg)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        sx = img.width / page_width
        sy = img.height / page_height

        for seg in risky_by_page.get(page_idx, []):
            x0, y0, x1, y1 = seg["bbox"]
            rect = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
            draw.rectangle(rect, fill=(255, 76, 76, 80), outline=(220, 20, 20, 180), width=2)

        combined = Image.alpha_composite(img, overlay)
        st.image(combined, caption=f"Page {page_idx + 1}", use_container_width=True)

    doc.close()


def main():
    st.title("PDF Risk Analyzer")
    st.caption("Upload a PDF, run the trained Legal-BERT model, and highlight risky contract text.")

    with st.sidebar:
        st.subheader("Model Settings")
        model_dir = st.text_input("Model directory", value="saved_cuad_legal_bert")
        force_cpu = st.checkbox("Force CPU", value=False)
        batch_size = st.slider("Batch size", min_value=2, max_value=64, value=16, step=2)
        max_length = st.slider("Max token length", min_value=128, max_value=512, value=320, step=32)

        st.subheader("Risk Rules")
        threshold = st.slider("Confidence threshold", min_value=0.30, max_value=0.99, value=0.55, step=0.01)
        min_chars = st.slider("Min block length", min_value=20, max_value=300, value=40, step=10)
        use_keywords = st.checkbox("Use keyword fallback", value=True)
        zoom = st.slider("Viewer zoom", min_value=1.0, max_value=3.0, value=1.6, step=0.2)

    try:
        tokenizer, model, device, labels = load_model(model_dir, force_cpu)
    except Exception as exc:
        st.error(f"Could not load model from '{model_dir}'. Error: {exc}")
        st.stop()

    st.success(f"Model loaded on: {device}")

    default_risky = suggest_risky_labels(labels)
    risky_labels = st.multiselect(
        "Labels considered risky",
        options=labels,
        default=default_risky,
        help="Predicted segments with one of these labels and confidence >= threshold will be highlighted.",
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is None:
        st.info("Upload a PDF to start analysis.")
        st.stop()

    pdf_bytes = uploaded_file.read()

    if st.button("Analyze PDF", type="primary"):
        with st.spinner("Processing document and scoring risk..."):
            segments = extract_blocks(pdf_bytes, min_chars=min_chars)
            scored = classify_segments(
                segments,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
            risky_segments = [
                seg for seg in scored if is_risky(seg, risky_labels, threshold, use_keywords)
            ]

        st.subheader("Summary")
        st.write(f"Total text blocks evaluated: {len(scored)}")
        st.write(f"Risky blocks highlighted: {len(risky_segments)}")

        if not risky_segments:
            st.success("No risky blocks found with current settings.")
            st.stop()

        preview_rows = []
        for seg in risky_segments[:200]:
            preview_rows.append(
                {
                    "page": seg["page"] + 1,
                    "label": seg["label"],
                    "confidence": round(seg["score"], 4),
                    "text": seg["text"][:300],
                }
            )

        st.subheader("Risky Content Table")
        st.dataframe(preview_rows, use_container_width=True)

        st.subheader("Highlighted PDF")
        render_highlighted_pages(pdf_bytes, risky_segments, zoom=zoom)


if __name__ == "__main__":
    main()
