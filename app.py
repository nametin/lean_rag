import streamlit as st
import pandas as pd
import numpy as np
import re
import openai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_all_models_and_data():
    df = pd.read_json("clean_lemmas_corpus.json.gz", lines=True, compression="gzip")

    # try:
        
    # except FileNotFoundError:
    #     st.error("KRİTİK HATA: `clean_lemmas_corpus.json.gz` dosyası bulunamadı. GitHub depona yüklediğinden emin ol.")
        # return None, None, None, None

    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    df['doc_text'] = df.apply(build_doc_text, axis=1) 
    
    name_texts = df['lemma_name'].fillna('').map(normalize_text).tolist() 
    name_tokens = [tokenize(t) for t in name_texts] 
    bm25_name = BM25Okapi(name_tokens)

    doc_texts = df['doc_text'].tolist()
    doc_tokens = [tokenize(t) for t in doc_texts]
    bm25_doc = BM25Okapi(doc_tokens)

    doc_embeddings = emb_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    return df, emb_model, bm25_name, bm25_doc, doc_embeddings


_punct_keep = set(['+', '*', '^'])
NOISE_WORDS = re.compile(r"(ModEq|pow_mod|_iff|inj|injective|gcd|dvd|div|mod|lt|multichoose|sub|_eq_|mono|strictMono|choose)", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r'\s+', ' ', s).strip()

def tokenize(s: str):
    toks, buf = [], []
    for ch in s:
        if ch in _punct_keep:
            if buf: toks.append(''.join(buf)); buf = []
            toks.append(ch)
        elif ch.isspace():
            if buf: toks.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf: toks.append(''.join(buf))
    return toks

def build_doc_text(row: pd.Series) -> str:
    nm = normalize_text(row.get('lemma_name', ''))
    st = normalize_text(row.get('statement', ''))
    cat = normalize_text(row.get('category', ''))
    name_boost = f"{nm} {nm} {nm}"
    return f"{name_boost} || {cat} || {st}"

def extract_goal_symbols(goal: str):
    g = goal or ""
    return {
        '+': ('+' in g), '*': ('*' in g), '^': ('^' in g),
        'succ': ('succ' in g.lower()), '0': ('0' in g), '1': ('1' in g),
    }

def build_query(goal: str):
    gnorm = normalize_text(goal)
    syms = extract_goal_symbols(gnorm)
    intent = []
    if re.search(r"\ba \+ b = b \+ a\b", gnorm) or "comm" in gnorm: intent += ["comm"]
    if re.search(r"\(a \+ b\) \+ c = a \+ \(b \+ c\)", gnorm) or "assoc" in gnorm: intent += ["assoc"]
    if re.search(r"\b0 \+ a = a\b", gnorm): intent += ["zero_add"]
    if re.search(r"\ba \+ 0 = a\b", gnorm): intent += ["add_zero"]
    if re.search(r"\b1 \* a = a\b", gnorm): intent += ["one_mul"]
    if re.search(r"\ba \* 1 = a\b", gnorm): intent += ["mul_one"]
    if "succ" in gnorm: intent += ["succ"]
    if '^' in gnorm:
        intent += ["pow"]
        if re.search(r"\^ 0\b", gnorm): intent += ["pow_zero","one_pow"]
        if re.search(r"\^\s*\(b \+ c\)", gnorm): intent += ["pow_add","mul_pow"]
    if syms['+']: intent += ['add','+']
    if syms['*']: intent += ['mul','*']
    if syms['^']: intent += ['^']
    if syms['0']: intent += ['0']
    if syms['1']: intent += ['1']
    q = (gnorm + ' ' + ' '.join(intent)).strip()
    return q, syms, intent

def bm25_scores(query: str, bm25_name, bm25_doc):
    t = tokenize(normalize_text(query))
    s_name = bm25_name.get_scores(t)
    s_doc = bm25_doc.get_scores(t)
    return 1.2*s_name + 1.0*s_doc

def emb_scores(query: str, emb_model, doc_embeddings):
    if emb_model is None or doc_embeddings is None:
        return np.zeros(len(doc_embeddings), dtype=np.float32)
    qv = emb_model.encode([normalize_text(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
    return (doc_embeddings @ qv)

def get_candidates(query: str, k, bm25_name, bm25_doc):
    s = bm25_scores(query, bm25_name, bm25_doc)
    idx = np.argsort(-s)[:k]
    return idx, s[idx]

def combine_and_rerank(goal: str, k, clean_lemmas, emb_model, bm25_name, bm25_doc, doc_embeddings):
    q, syms, intent = build_query(goal)
    idx_bm25, bm25_top = get_candidates(q, k=k, bm25_name=bm25_name, bm25_doc=bm25_doc)
    base_scores = np.zeros(len(clean_lemmas), dtype=np.float32)
    base_scores[idx_bm25] += bm25_top.astype(np.float32)

    if emb_model is not None and doc_embeddings is not None:
        s_emb = emb_scores(q, emb_model, doc_embeddings)
        base_scores += 0.4 * s_emb

    rows = []
    order = np.argsort(-base_scores)[:max(k, 100)]
    for i in order:
        row = clean_lemmas.iloc[i]
        name = str(row['lemma_name'])
        stmt = str(row['statement'])
        cat = str(row.get('category',''))
        syml = set(row.get('symbols', [])) if isinstance(row.get('symbols', []), list) else set()
        lname = name.split('.')[-1].lower()

        if not syms['^'] and cat == 'pow_*': continue
        is_pow_cat = (cat == 'pow_*')
        if syms['+'] and ('+' not in syml) and not is_pow_cat: continue
        if syms['*'] and ('*' not in syml) and not is_pow_cat: continue
        if syms['succ'] and ('succ' not in syml) and ('succ' not in name.lower()): continue
        
        fam_ok = True
        if syms['+'] and not (cat.startswith('add') or 'add' in name.lower() or cat == 'zero_add_or_add_zero' or 'succ' in name.lower()) and not is_pow_cat: fam_ok = False
        if syms['*'] and not (cat.startswith('mul') or 'mul' in name.lower()) and not is_pow_cat: fam_ok = False
        if syms['^'] and not (cat == 'pow_*' or '^' in stmt): fam_ok = False
        if not fam_ok: continue

        S_b = float(base_scores[i])
        if name.startswith("Nat."): S_b *= 1.2
        else: S_b *= 0.9

        S_sym = 0.0
        if syms['+'] and '+' in syml: S_sym += 0.6
        if syms['*'] and '*' in syml: S_sym += 0.6
        if syms['^'] and '^' in syml: S_sym += 0.8
        if syms['succ'] and ('succ' in syml or 'succ' in name.lower()): S_sym += 0.5

        S_cat = 0.0
        for kw in ['comm','assoc','zero','one','succ','pow','add','mul']:
            if kw in lname: S_cat += (0.6 if kw in {'comm','assoc','zero','one','succ','pow'} else 0.3)

        S_intent = 0.0
        specific_intents = ['comm', 'assoc', 'zero_add', 'add_zero', 'one_mul', 'mul_one', 'succ', 'pow_add', 'pow_zero']
        for kw in specific_intents:
            if kw in intent and kw in lname: S_intent += 5.0
        
        S_neg = 0.0
        if NOISE_WORDS.search(name): S_neg = -10.0; continue
        if NOISE_WORDS.search(stmt): S_neg = -10.0; continue
        if lname.endswith('_def'): S_neg = -10.0; continue

        S_len = max(0.0, 1.0 - min(len(stmt), 250)/250.0) * 0.5
        S_total = S_b + S_sym + S_cat + S_len + S_intent

        rows.append({
            'lemma_name': name, 'category': cat, 'statement': stmt,
            'source_ref': row['source_ref'], 'S_total': S_total,
        })

    if not rows:
        return pd.DataFrame(columns=['lemma_name', 'category', 'statement', 'source_ref', 'S_total'])
    
    out = pd.DataFrame(rows).sort_values('S_total', ascending=False).reset_index(drop=True)
    return out

def retrieve_top3_app(goal: str, clean_lemmas, emb_model, bm25_name, bm25_doc, doc_embeddings):
    scored = combine_and_rerank(goal, k=50, clean_lemmas=clean_lemmas, emb_model=emb_model, 
                                bm25_name=bm25_name, bm25_doc=bm25_doc, 
                                doc_embeddings=doc_embeddings)
    cols = ['lemma_name', 'category', 'statement', 'source_ref']
    final_cols_to_return = [col for col in cols if col in scored.columns]
    return scored.head(3)[final_cols_to_return]


def format_context_for_prompt(df_top3: pd.DataFrame) -> str:
    if df_top3.empty:
        return "Üzgünüm, bu hedef için korpusta (bilgi bankamda) uygun bir lemma bulamadım."
    context_str = "Kullanıcının hedefine uygulanabilecek en alakalı lemmalar:\n\n"
    for _, row in df_top3.iterrows():
        context_str += f"- Lemma Adı: {row['lemma_name']}\n"
        context_str += f"  - Statement (İfade): `{row['statement']}`\n"
        context_str += f"  - Kategori: {row['category']}\n"
        context_str += f"  - Kaynak: {row['source_ref']}\n\n"
    return context_str

def build_rag_prompt_openai(goal: str, context_str: str) -> list[dict]:
    system_prompt = """
    Sen, Lean/Mathlib matematik kütüphanesi konusunda uzman bir asistansın.
    Görevin, kullanıcının verdiği 'Kullanıcı Hedefi' (bir eşitlik) için en iyi 
    'ilk hamle' olabilecek lemmaları önermektir.
    Sana bu hedef için bilgi bankamdan bulduğum en alakalı lemmaları 'Bağlam (Context)' 
    olarak vereceğim.
    Cevabın kısa, profesyonel ve bir asistan gibi olmalı. 'Bağlam (Context)' 
    içindeki bilgileri kullanarak bir öneride bulun. En iyi adayı vurgula.
    Lemma adlarını (örn: `Nat.add_comm`) `code snippet` olarak formatla.
    Eğer 'Bağlam (Context)' 'uygun bir lemma bulamadım' diyorsa, bunu kibarca ilet 
    ve (eğer biliyorsan) genel bir öneride bulun.
    """
    user_prompt = f"""
    --- BAĞLAM (Context) ---
    {context_str}
    
    --- KULLANICI HEDEFİ ---
    {goal}
    
    --- GÖREV ---
    Yukarıdaki 'Bağlam (Context)' bilgilerine dayanarak, 'Kullanıcı Hedefi'ni 
    kanıtlamak için en iyi ilk hamle önerini yap.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def get_rag_suggestion_openai(goal: str, client, clean_lemmas, emb_model, bm25_name, bm25_doc, doc_embeddings) -> str:

    try:
        df_top3 = retrieve_top3_app(goal, clean_lemmas, emb_model, bm25_name, bm25_doc, doc_embeddings)
        
        context_str = format_context_for_prompt(df_top3)
        messages = build_rag_prompt_openai(goal, context_str)
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"OpenAI RAG zincirinde hata oluştu: {e}")
        return "OpenAI RAG zincirinde hata oluştu: {e}"


st.set_page_config(page_title="Lean Lemma Önerici", layout="centered")
st.title("Basic Lean/Mathlib Lemma Öneri Sistemi")
st.markdown("Verdiğiniz `Nat` eşitlik hedefini kanıtlamak için en iyi ilk hamle olabilecek lemmayı önerir. Dataset yalnızca basic lemma'ları içerdiği için (a+b) = a+b veya a*b = b*a şeklindeki lemma'lar ile deneyiniz." )

data_load_state = st.text("Modeller ve dataset yükleniyor...")
try:
    (clean_lemmas, emb_model, bm25_name, 
     bm25_doc, doc_embeddings) = load_all_models_and_data()
    data_load_state.text("Modeller ve dataset hazır!".format(len(clean_lemmas)))
except Exception as e:
    st.error(f"model veya dataset hatası")
    st.stop() 

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("api key bulnuamadı")
    st.stop()

with st.form(key="goal_form"):
    user_goal = st.text_input(
        label="Kanıt Hedefinizi Girin:", 
        placeholder="Örn: a + b = b + a"
    )
    submit_button = st.form_submit_button(label="Lemma Önerisi Getir")

if submit_button and user_goal:
    with st.spinner("RAG sistemi çalışıyor... (Retriever arıyor, GPT-4o düşünüyor...)"):
        try:
            suggestion = get_rag_suggestion_openai(
                user_goal, client, clean_lemmas, emb_model, 
                bm25_name, bm25_doc, doc_embeddings
            )
            st.markdown(suggestion)
        except Exception as e:
            st.error(f"Öneri alınırken bir hata oluştu: {e}")
elif submit_button and not user_goal:
    st.warning("Lütfen bir hedef girin.")