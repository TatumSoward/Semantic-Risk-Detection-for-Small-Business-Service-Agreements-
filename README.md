# Semantic-Risk-Detection-for-Small-Business-Service-Agreements-
Group 9 AIG230 - NLP Project. Using Legal-BERT to perform Automated Clause Extraction and Risk Scoring on contracts/legal agreements.
## Project Description
### Problem
Freelancers and small business owners may not have the necessary legal background to identify predatory clauses in standard terms of service or service-level agreements, which can expose them to unfair liability shifts, hidden auto-renewal fees, or excessive intellectual property claims. An automated, context-aware tool that can identify and highlight the risk to non-experts, can save these users money in unwanted legal fees.
### Solution
We will use Legal-BERT (a transformer model pre-trained on legal corpora) to perform Automated Clause Extraction and Risk Scoring.
### Dataset
<b> Contract Understanding Atticus Dataset (CUAD): </b> https://www.atticusprojectai.org/cuad <br>
A database of legal contracts and terms of service. With 13,000 labelled clauses and 41 labels.
### Key NLP Concepts
* <b> Token Classification: </b> To identify specific functional semantics spans.
* <b> Cross-Encoder: </b> Compare clauses with a fair example using FAISS or ChromaDB. Use Cosine Similarity and Natural Language Interface to handle contradictory words like “not”.
* <b> Integrated Gradients: </b> To highlight specific phrases within the clause that contributed most to the high-risk score, providing a Plain English explanation of the potential threat.
### Objectives
To achieve a <b>Precision and Recall score of 0.88+</b> on identifying high-risk clauses within unseen contracts. To increase the efficiency of reviewing and transparency of ToS for users. 
### Deliverables
* <b> Web-Based Interface: </b> A React/Streamlit dashboard where users can upload PDF/Word contracts.
* <b> A Classification Model using Legal-BERT: </b> Code located on GitHub.
* <b> Heatmap: </b> Using the Attribution score to map the risk back onto sentences.
* <b> Demo: </b> Where we demonstrate on our Web Application:
  1. Upload an example document.
  2. Have the model identify and extract predatory clauses.
  3. Highlight the risky clause in the document.
## Schedule
<b>Week 1-2 </b>Code a functioning clause classification model using Legal-BERT and CUAD ✅<br>
<b>Week 3 </b>Implementing FAISS comparison, flagging problematic clauses ✅<br>
<b>Week 4 </b>Build an UI with document uploader and highlighted text output using Streamlit ❌<br>
<b>Week 5 </b>Tweak and refine to hit 0.88 Precision metric ❌<br>
<b>Week 6 </b>Demo prep, slides, video, ensuring that the demo will work the day of ❌<br>
## Progress (as of 2026-03-02)
### Completed
* <b>cuad_clause_classification.ipynb:</b> Test code for the set-up of Legal-BERT.
  * Classifies the meaning of sentences in legal documents.
* <b>semantic_risk_detection.ipynb:</b> Test code for FAISS, Integrated Gradient, and dashboard.
  * ID's the risk of each document sentence.
  * Has FAISS Cross Encoder.
  * Has Integrated Gradient Heatmap.
  * Has first progress for a dashboard.
  * Beginning of a full pipeline.
* <b>version1.ipynb:</b> First version of a full pipeline.
* <b>NLP_Project_Code.ipynb:</b> Cleaned up version of version1.ipynb, it contains multiple fine-tuned versions of the pipeline to test performance.
### To Do
* Save preprocessing pipeline and model as .pkl files and deploy on the cloud.
* Create a UI to upload .pdf to the app using Streamlit.
* Have an output that highlights risky sentences in the original location in a .pdf file.
* Play with hyperparameters to improve model performance to 0.88 Precision/Recall goal.
