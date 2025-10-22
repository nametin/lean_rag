# Lean/Mathlib için Goal-Aware Lemma Öneri Sistemi (RAG)

## 1. Amaç

Lean proof assistant (bir fonksiyonel programlama dili ve interactive theorem prover) ve onun matematik kütüphanesi olan Mathlib kütüphanesinden seçtiğim küçük bir corpus üzerinde, kullanıcı tarafından verilen bir `Nat` eşitlik hedefine yönelik "ilk hamle" olabilecek en iyi 3 lemmayı öneren bir RAG sistemidir.

## 2. Dataset

Bu proje için temel veri kaynağı olarak Hugging Face üzerinde bulunan `tasksource/leandojo` veri seti kullanılmıştır.

Ancak, `leandojo` tüm Mathlib kütüphanesini içerir. Projenin amacına (basit `Nat` eşitlikleri) uygun, odaklanmış ve gürültüden arındırılmış bir RAG sistemi kurabilmek için bu ham veri seti üzerinde bir temizleme süreci uygulandı:

1.  **Kapsam Belirleme:**
    * Veri seti öncelikle sadece `Mathlib/Data/Nat/Basic.lean`, `Mathlib/Data/Nat/Pow.lean` ve `Init/Data/Nat/Basic.lean` (canonical tanımlar için) dosyalarından gelen `Nat` lemmalarını içerecek şekilde filtrelenmiştir.

2.  **Gürültü Ayıklama:**
    * **Sadece Eşitlikler:** Korpusa sadece `a = b` formatındaki *koşulsuz eşitlik* ifadeleri dahil edilmiştir. `≤`, `<`, `∣` bölünebilirlik veya eşitsizlik veya diğer ilişkileri içeren lemmalar ayıklanmıştır.
    * **Yasaklı Token:** Proje kapsamı dışındaki `gcd`, `cast`, `Int.cast`, `ℤ` , `lt` , `sub`, `mono` veya `multichoose` gibi karmaşık tokenları içeren tüm lemmalar veri setinden çıkarılmıştır.

3.  **Canlı Veri Doğrulama ve Zenginleştirme:**
    * `leandojo` veri setindeki `statement` metinlerinin her zaman güncel veya doğru formatta olmamasından dolayı, her bir potansiyel lemma için `robust_extract_statement` adında özel bir fonksiyon kullanılmıştır.
    * Bu fonksiyon, lemmanın `commit` ve `file_path` bilgilerini kullanarak doğrudan GitHub'daki Lean/Mathlib reposuna canlı bir network çağrısı yapmış; lemmanın en güncel, ham kodunu çekmiş, `:=` veya `by` gibi kısımları ayıklayarak temiz bir `statement` elde etmiştir.
    * Bu adım, veri setindeki `statement`'ların doğruluğunu ve temizliğini garanti altına almıştır.

Bu preprocessing sonucunda, binlerce aday arasından seçilen, hedefe yönelik **~40 adet lemma** içeren bir "mini-korpus" (`clean_lemmas_corpus.json.gz`) oluşturulmuştur. Zaman kısıtlamasından ötürü, dataseti daha fazla büyütemedim...

## 3. Çözüm Mimarisi:

Sistem bir RAG mimarisi kullanır:

### 3.1. (R) Retrieval: Hibrit Arama ve Filtreleme

Kullanıcının sorgusu (`goal`), özel olarak korpusta aranır:

1.  **Hybrid Search:** Aday lemmaları bulmak için iki farklı skor birleştirilir:
    * **BM25 (Kelime Bazlı):** `rank_bm25` kütüphanesi kullanılarak sorgu metni ile lemma adı/statement'ı arasındaki anahtar kelime eşleşmesi puanlanır.
    * **Vektör Benzerliği (Semantik):** `sentence-transformers/all-MiniLM-L6-v2` modeli kullanılarak sorgu ile lemmalar arasındaki anlamsal (semantik) yakınlık puanlanır.

2.  **Aşırı Filtreleme (Aggressive Reranking):**
    * Bu adaylar daha sonra `NOISE_WORDS` listesi (`_def`, `lt`, `choose` vb.) ve eleme kriterleri (örn: sorguda `*` yoksa `pow` önerme) kullanılarak elenir.
    * Kalan adaylar, sorgudaki "niyete" (`comm`, `assoc`, `zero_add` gibi) ne kadar uyduklarına göre bonus (`S_intent`) alırlar.
    * En yüksek `S_total` skoruna sahip Top-3 lemma LLM'e verilmek üzere seçilir.

### 3.2. (A) Augmentation: Dinamik Prompt Oluşturma

Top-3 lemma, OpenAI'ye gönderilmek üzere bir context metnine dönüştürülür. Eğer "R" adımı hiçbir lemma bulamazsa, context metni olarak "Üzgünüm, bu hedef için korpusta uygun bir lemma bulamadım." mesajı oluşturulur.

### 3.3. (G) Generation: OpenAI (gpt-4o) ile Akıl Yürütme

Oluşturulan bu context metni ve kullanıcının orijinal hedefi, `gpt-4o` modeline bir `system_prompt' ile birlikte gönderilir.

## 4. Elde Edilen Sonuçlar

* **Retriever Başarısı (Hit@3):** Geliştirme sırasında kullanılan 12 temel hedef sorgusundan oluşan `eval_df` test setinde, "R" (Retrieval) adımı **%75 (12'de 9)** oranında doğru lemmayı ilk 3'te bulmayı başarmıştır.
* **RAG Başarısı:** "R" adımının başarısız olduğu %25'lik dilimde:
    * **TEST 2 (`a + 0 = a`):** "R" adımı yanlış lemmalar (`Nat.add_comm`) getirse de, `gpt-4o` bu yanlış context'i kullanarak 2 adımlı geçerli bir çözüm yolu üretebilmiştir.
    * **TEST 3 (`a ^ 0 = 1`):** "R" adımı boş dönse de, `gpt-4o` kullanıcıya `...ancak, genellikle ... pow_zero lemması kullanılır.` şeklinde bir öneride bulunmuştur.

## 5. Canlı Demo
Oluşturduğum web arayüzünü public olarak paylaşamıyorum çünkü her sorguda openai hesabımdan kredi yiyor. Bunun yerine maille iletebilirim. ahmet.metin@sabanciuniv.edu 
Veya 6. bölümdeki şekilde yerel çalıştırabilirsiniz.

## 6. Lokal Kurulum
git clone https://github.com/nametin/lean_rag.git
cd lean_rag
pip install -r requirements.txt

< Proje kökünde .streamlit adında yeni bir klasör oluşturun. Klasörün içerisinde secrets.toml adında bir dosya oluşturun.
secrets.toml dosyasının içeriğine OPENAI_API_KEY = "key" şeklinde keyinizi girin. >

streamlit run app.py
