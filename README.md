# Lean-RAG: Goal-Aware Lemma Suggestion for Lean

Lean-RAG is a specialized Retrieval-Augmented Generation (RAG) system designed to assist in Automated Theorem Proving. It acts as a "first-move" assistant by suggesting the top-3 most relevant lemmas from the Mathlib library for a given Nat (Natural Number) equality goal.

## 1. Objective

The system targets users working with the Lean 4 proof assistant. Using a curated corpus derived from Mathlib, it analyzes a user's proof goal (e.g., a + 0 = a) and retrieves the most applicable equality lemmas to advance the proof state.

## 2. Dataset & Curation
The project utilizes the tasksource/leandojo dataset from Hugging Face as the base source. However, raw Mathlib data is often too noisy for a targeted RAG application. A rigorous cleaning pipeline was implemented to create a focused "Mini-Corpus":

1.  **Scope Restriction:**
    * Filtered to include lemmas strictly from Mathlib/Data/Nat/Basic.lean, Mathlib/Data/Nat/Pow.lean, and Init/Data/Nat/Basic.lean to focus on canonical definitions.
      
2.  **Noise Reduction:**
    * **Equalities Only:** The corpus is restricted to unconditional equality statements (a = b). Complex relations like ≤, <, or ∣ were filtered out.
    * **Token Ban List:** Lemmas containing out-of-scope or complex tokens (e.g., gcd, cast, Int.cast, ℤ, multichoose) were excluded to maintain domain focus.

3.  **Live Data Validation:**
    * Since static dataset dumps can be outdated or malformatted, a custom robust_extract_statement function was implemented.
    * This function uses the lemma's metadata (commit hash and file path) to make live API calls to the GitHub repository, fetching the raw source code. It then parses the clean statement, removing syntax noise (like := or by blocks).
    * This preprocessing pipeline yielded a high-precision knowledge base of approximately 40 verified lemmas, stored in clean_lemmas_corpus.json.gz to serve as a noise-free ground truth for the retrieval module.

## 3. System Architecture:

The solution implements a classic RAG pipeline with a specialized retrieval strategy:

### 3.1. (R) Retrieval: Hybrid Search & Aggressive Reranking

The user's goal is processed through a multi-stage pipeline:

1.  **Hybrid Search:** Candidates are retrieved using a weighted combination of:
    * **Sparse Retrieval (BM25):** Matches exact mathematical notation and keywords.
    * **Dense Retrieval (Semantic):** Uses sentence-transformers/all-MiniLM-L6-v2 to capture structural similarities.

2.  **Aggressive Filtering & Reranking:**
    * Candidates containing NOISE_WORDS (e.g., internal definitions like _def) or irrelevant operations (e.g., suggesting pow lemmas if the goal has no exponentiation) are discarded.
    * Surviving candidates receive a bonus score (S_intent) if they align with the goal's structural intent (e.g., commutativity, associativity, identity properties).
    * The top-3 lemmas by S_total are passed to the next stage.

### 3.2. (A) Augmentation:

The top-3 retrieved lemmas are formatted into a structured context block. If the retrieval stage returns no valid candidates, a fallback context (Eng: "No specific lemma found in corpus...") is generated to prevent hallucination, allowing the LLM to rely on its parametric knowledge.
### 3.3. (G) Generation:

The augmented context and the user's original goal are fed into OpenAI GPT-4o via a custom system prompt designed to perform logical reasoning and suggest the next proof step.

## 4. Results

* **Retrieval Performance (Hit@3):** On a specialized test set of 12 distinct equality goals (eval_df), the retrieval module achieved a 75% Success Rate (9/12) in placing the correct lemma within the top 3 results.
* **RAG Robustness:** (In cases where retrieval failed)
    * **TEST 2 (`a + 0 = a`):** Even when the retrieval stage returned irrelevant lemmas (e.g., `Nat.add_comm`), `gpt-4o` was able to handle the noisy context and generate a valid 2-step proof path.
    * **TEST 3 (`a ^ 0 = 1`):** When the retrieval stage returned empty results, `gpt-4o` successfully fell back on its parametric knowledge, suggesting to the user: `...however, the pow_zero lemma is typically used here.`
  
## 5. Demo
Due to API costs, a public live demo is currently unavailable. However, I can provide access to the live deployment upon request via ahmet.metin@sabanciuniv.edu
OR You can run the application locally (see Section 6).

### An example screenshot
![Image](https://github.com/user-attachments/assets/f5a05f09-9f5d-402d-9070-ade98d5fcdcc) 

## 6. Local Installation
git clone https://github.com/nametin/lean_rag.git

cd lean_rag

pip install -r requirements.txt

<Create a .streamlit folder in the project root. Inside, create a secrets.toml file. Add your API key: OPENAI_API_KEY = "key" >

streamlit run app.py
