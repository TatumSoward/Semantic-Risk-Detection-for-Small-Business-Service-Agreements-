# Streamlit UI for Version 3 Model

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Make sure your trained model is available

The app expects a Hugging Face model folder (with files like `config.json`, tokenizer files, and model weights).

Default model folder in the app:

- `saved_cuad_legal_bert`

If your model is in a different folder, set the path in the sidebar `Model directory` field.

## 3. Run Streamlit

```bash
streamlit run streamlit_app.py
```

## 4. Use the app

1. Upload a PDF.
2. Click **Analyze PDF**.
3. Wait for processing (loading spinner).
4. Review:
   - Risky Content Table
   - Highlighted pages in the PDF viewer panel

## Notes

- Risk highlighting is based on model-predicted label + confidence threshold.
- You can control threshold, token length, and selected risky labels from the sidebar.
- `Use keyword fallback` can highlight additional risky snippets by legal-risk keywords.
