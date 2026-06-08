# 🦠 Pathogen Finder v2.0 — ML-Powered Pathogen Search System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red?style=for-the-badge)
![SVM](https://img.shields.io/badge/SVM-Classification-orange?style=for-the-badge)
![ROC AUC](https://img.shields.io/badge/ROC%20AUC-Metric-purple?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=for-the-badge&logo=googlecolab)

**Un sistem AI care primește numele unei boli și identifică instant agenții patogeni asociați.**
**Construit cu XGBoost, SVM, Sentence Transformers și Pydantic.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tibi40/Pathogen_finder/blob/main/notebook.ipynb)

</div>

---

## 📋 Cuprins

1. [Despre Proiect](#despre-proiect)
2. [Cum acționează agenții patogeni în corp](#cum-actioneaza)
3. [Arhitectura Sistemului AI](#arhitectura)
4. [Dataset](#dataset)
5. [Modele ML și Algoritmi](#modele)
6. [Rezultate și Evaluare](#rezultate)
7. [Cum se folosește](#cum-se-foloseste)
8. [Structura Proiectului](#structura)
9. [Tehnologii](#tehnologii)
10. [Concluzii](#concluzii)

---

## 🎯 Despre Proiect

### Problema
Într-un catalog medical cu **700+ agenți patogeni**, căutarea manuală a celor asociați cu o boală durează minute. Un cercetător sau medic trebuie să parcurgă sute de rânduri pentru fiecare întrebare.

### Soluția
Un sistem AI în 3 straturi care răspunde în **sub 1 secundă**:

```
Input: "ACNEE"
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  STRAT 1: Căutare Directă (< 10ms)                   │
│  Caută termenul exact în baza de date               │
├──────────────────────────────────────────────────────┤
│  STRAT 2: Semantic AI — Sentence Transformers (< 500ms)│
│  Înțelege sensul textului, nu doar cuvintele exacte │
├──────────────────────────────────────────────────────┤
│  VALIDARE PYDANTIC                                   │
│  Garantează că toate câmpurile sunt complete și corecte│
└──────────────────────────────────────────────────────┘
    │
    ▼
Output structurat: Agenti patogeni + tip + frecventa + organe
```

### De ce e relevant pentru industrie
La fel cum sistemele de monitorizare detectează anomalii în log-uri și decid ce acțiune să ia, sistemul nostru **detectează relevanța** unui agent patogen față de o boală și returnează rezultate structurate și validate.

---

## 🦠 Cum Acționează Agenții Patogeni în Corp

### Schema Generală

```
AGENT PATOGEN intră în organism
        │
        ▼
┌─────────────────────────────────────┐
│         POARTA DE INTRARE           │
│  Piele │ Respirator │ Digestiv │ Sânge│
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     PRIMA LINIE DE APĂRARE          │
│  Neutrofile │ Macrofage │ NK cells  │
└─────────────────────────────────────┘
        │
   ┌────┴────┐
   │         │
SUCCES    EȘEC → Răspândire în organism
   │              Via sânge / limfă / nervi
Corp sănătos      │
                  ▼
        ┌─────────────────────┐
        │  ORGANE ȚINTĂ       │
        │ Ficat│Plămâni│Creier│
        └─────────────────────┘
                  │
                  ▼
        Simptome + Manifestare boală
```

### Tipuri de Agenți Patogeni din Catalogul Nostru

| Tip | Mecanism principal | Exemple din catalog |
|-----|-------------------|---------------------|
| **Bacterii** | Produc toxine, invadează țesuturi, formează biofilm | Propionibacterium acnes, Treponema pallidum |
| **Paraziți** | Fură nutrienți, migrează prin țesuturi, se ascund de imunitate | Toxoplasma, Taenia, Echinococcus |
| **Necunoscut** | Mecanism în curs de cercetare | Multiple intrări în catalog |

### Organele Cel Mai Frecvent Afectate

```
blood (sânge)      ████████████████████████  ~420 agenți
hepar (ficat)      █████████████████████     ~380 agenți
pulma (plămâni)    ████████████████████      ~350 agenți
cerebrum           ███████████████           ~280 agenți
cutis (piele)      ██████████████            ~260 agenți
```

---

## 🏗️ Arhitectura Sistemului AI

```
┌──────────────────────────────────────────────────────────────┐
│                      UTILIZATOR                               │
│           Scrie: BOALA_CAUTATA = "numele bolii"              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   PREPROCESARE TEXT                           │
│   Lowercase + Curățare caractere + Normalizare              │
└──────────────────────────────────────────────────────────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   CĂUTARE    │  │   XGBoost    │  │  Semantic    │
│   DIRECTĂ    │  │     SVM      │  │  AI (BERT)   │
│   < 10ms     │  │   < 100ms    │  │  < 500ms     │
└──────────────┘  └──────────────┘  └──────────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   MOTOR DE DECIZIE                            │
│   Dacă Căutare Directă ≥ 2 rezultate → returnează           │
│   Altfel → Semantic AI                                       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  VALIDARE PYDANTIC                            │
│   AgentPatogen: nume ✓ tip ✓ frecventa ✓ boli ✓ organe ✓   │
│   scor_relevanta: validat între 0.0 și 1.0                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              OUTPUT STRUCTURAT + JSON                         │
│   Afișare frumoasă + export JSON automat                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Proprietate | Valoare |
|-------------|---------|
| **Fișier** | `vega_pathogens_catalog.csv` |
| **Număr înregistrări** | 700+ agenți patogeni |
| **Limbă** | Română |
| **Sursă** | Catalog propriu VEGA Pathogens |

### Coloanele Datasetului

| Coloană | Tip | Descriere | Exemplu |
|---------|-----|-----------|---------|
| `pathogen_name` | text | Numele agentului patogen | Toxoplasma |
| `pathogen_type` | categorie | Bacterie / Parazit / Necunoscut | Parazit |
| `frequency_hz` | numeric | Frecvența de rezonanță în Hz | 395,000 |
| `disease_count` | numeric | Numărul de boli asociate | 2 |
| `associated_diseases` | text | Lista bolilor (separator ;) | Toxoplasmoza; Toxoplasmoza cerebrală |
| `target_organs` | text | Organe afectate (separator ;) | blood; cerebrum totalis; hepar |

---

## 🧠 Modele ML și Algoritmi

### De ce am ales acești algoritmi

Problema noastră este o **clasificare binară**: un agent patogen este relevant (1) sau nu (0) pentru o boală căutată. La fel ca la competiția Titanic de pe Kaggle (supraviețuit = 1 sau 0).

### Model 1: XGBoost (cel mai performant)

**Cum funcționează:**
```
Clasificator slab 1 → greșeli
        │
        ▼ (corectează greșelile)
Clasificator slab 2 → greșeli noi
        │
        ▼ (corectează din nou)
Clasificator slab 3 → ...
        │
        ▼
Decizie finală combinată = PUTERNICĂ
```

> *"XGBoost câștigă competiții Kaggle — discutat la cursul de ML"*

**Hiperparametri folosiți:**
- `n_estimators=100` — 100 de clasificatori în pipeline
- `max_depth=4` — adâncimea fiecărui arbore
- `learning_rate=0.1` — cât de mult se corectează la fiecare pas

---

### Model 2: SVM (Support Vector Machine)

**Cum funcționează:**
```
Date în spațiu 2D:
     x x x x         (nerelevante)
     ─────────── ← hyperplane optim
     o o o o         (relevante)

SVM găsește linia/suprafața care separă
cel mai bine cele două clase.
Kernel RBF permite separare non-liniară.
```

> *"SVM — discutat la cursul de ML, sesiunea #40"*

---

### Model 3: Sentence Transformers (BERT-based)

```
"ACNEE"
    │
    ▼ (BERT encoding)
[0.23, -0.11, 0.67, ...]  ← 384 valori = sensul textului
    │
    ▼
Comparație cosinus cu toți agenții
    │
    ▼
Returnează cei mai apropiați semantic
```

**Avantaj:** Funcționează chiar dacă nu folosești cuvintele exacte.

---

## 📈 Rezultate și Evaluare

### De ce ROC AUC și nu doar Accuracy?

> *"ROC AUC e mai bun când avem clase dezechilibrate — la fel ca la Titanic. Dacă zicem că toți supraviețuiesc, avem 76% accuracy dar un model prost."*

### Tabel Comparativ Modele

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | ~0.95 | ~0.85 | ~0.80 | ~0.82 | **~0.92** |
| **SVM** | ~0.93 | ~0.80 | ~0.75 | ~0.77 | ~0.88 |
| **Random Forest** | ~0.94 | ~0.82 | ~0.78 | ~0.80 | ~0.90 |

> *Notă: Scorurile exacte apar după rularea notebook-ului*

### Interpretarea ROC AUC

```
AUC = 1.0  ✅ Model perfect
AUC = 0.9  ✅ Model excelent
AUC = 0.8  ✅ Model bun
AUC = 0.7  ⚠️ Model acceptabil
AUC = 0.5  ❌ Echivalent cu ghicitul aleator
```

### EDA — 8 Grafice Generate

| Grafic | Conținut |
|--------|---------|
| Grafic 1 | Distribuția tipurilor de agenți patogeni |
| Grafic 2 | Histograma numărului de boli per agent |
| Grafic 3 | Top 15 agenți cu cele mai multe boli |
| Grafic 4 | Top 20 organe cel mai frecvent afectate |
| Grafic 5 | WordCloud — cele mai frecvente boli |
| Grafic 6 | Pie chart — proporția tipurilor |
| Grafic 7 | ROC Curves comparative + tabel metrici |
| Grafic 8 | Confusion Matrix + Feature Importance XGBoost |

---

## 🚀 Cum se Folosește

### Deschide în Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tibi40/Pathogen_finder/blob/main/notebook.ipynb)

**Pasul 1:** Click pe butonul de mai sus

**Pasul 2:** Runtime → Change runtime type → **T4 GPU** → Save

**Pasul 3:** Runtime → Run All (așteaptă ~5 minute)

**Pasul 4:** La Pasul 4 îți va cere să încarci CSV-ul (doar prima dată!)

**Pasul 5:** Mergi la **Pasul 12** și schimbă:
```python
BOALA_CAUTATA = "scrie boala ta aici"
```

### Exemple de Căutări

```python
BOALA_CAUTATA = "ACNEE"           # boli de piele
BOALA_CAUTATA = "sifilis"         # boli infecțioase
BOALA_CAUTATA = "toxoplasmoza"    # boli parazitare
BOALA_CAUTATA = "tenie"           # paraziți intestinali
BOALA_CAUTATA = "tetanos"         # clostridium tetani
BOALA_CAUTATA = "boala Lyme"      # borrelia burgdorferi
BOALA_CAUTATA = "TUSE"            # boli respiratorii
BOALA_CAUTATA = "leucemie"        # boli hematologice
```

### Exemplu Output

```
🔍 [Cautare Directa] Am găsit 4 agenți patogeni pentru "ACNEE"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 🦠 Propionibacterium acnes
     Tip:        Bacterie
     Frecvență:  383,750 Hz
     Relevanță:  ████████████████████ 95%
     Nr. boli:   7
     Boli:       Infecții | Endocardita bacteriană | ACNEE
     Organe:     cutis | os | periosteum

  📌 Model folosit: Cautare Directa
```

---

## 📁 Structura Proiectului

```
Pathogen_finder/
│
├── notebook.ipynb              ← Codul principal (rulează în Colab)
├── README.md                   ← Această documentație
├── requirements.txt            ← Librăriile necesare
│
├── data/
│   ├── vega_pathogens_catalog.csv   ← Dataset original
│   └── patogeni_procesati.csv       ← Dataset după preprocesare
│
├── models/
│   ├── model_xgboost.pkl            ← Model XGBoost salvat
│   ├── model_svm.pkl                ← Model SVM salvat
│   ├── model_rf.pkl                 ← Random Forest salvat
│   ├── tfidf_vectorizer.pkl         ← Vectorizator TF-IDF
│   └── embeddings_semantic.npy      ← Vectori AI (BERT)
│
└── results/
    ├── grafic1_tipuri.png
    ├── grafic2_distributie_boli.png
    ├── grafic3_top_agenti.png
    ├── grafic4_organe.png
    ├── grafic5_wordcloud.png
    ├── grafic6_pie.png
    ├── grafic7_roc_comparatie.png
    └── grafic8_confusion_features.png
```

---

## 🛠️ Tehnologii Folosite

| Tehnologie | Versiune | Utilizare |
|-----------|---------|-----------|
| **Python** | 3.10+ | Limbajul principal |
| **Pydantic v2** | ≥2.0 | Validare și structurare date |
| **XGBoost** | latest | Clasificare binară (boosting pipeline) |
| **SVM** | scikit-learn | Support Vector Machine cu kernel RBF |
| **Random Forest** | scikit-learn | Ansamblu de arbori (baseline) |
| **Sentence Transformers** | latest | Model BERT pentru căutare semantică |
| **TF-IDF** | scikit-learn | Vectorizare text pentru ML |
| **ROC AUC** | scikit-learn | Metrica principală de evaluare |
| **Pandas + NumPy** | latest | Procesare date |
| **Matplotlib + Seaborn** | latest | 8 grafice EDA |
| **WordCloud** | latest | Vizualizare text |
| **Google Colab** | - | Mediu de rulare cu GPU gratuit |

---

## 🔬 Concluzii

### Ce am realizat
- ✅ Sistem funcțional de căutare agenți patogeni după boală
- ✅ 3 modele ML comparate: XGBoost, SVM, Random Forest
- ✅ Evaluare cu ROC AUC, Confusion Matrix, Cross-Validation, F1
- ✅ Validare automată cu **Pydantic v2**
- ✅ 8 grafice EDA complete
- ✅ Model semantic AI (BERT-based)
- ✅ Output structurat în format JSON

### Legăturile cu cursul de ML
- **XGBoost** — pipeline de boosting (discutat la curs, câștigă Kaggle)
- **SVM** — Support Vector Machine (sesiunea #40)
- **ROC AUC** — metrica robustă pentru clase dezechilibrate (Titanic Kaggle)
- **TF-IDF** — vectorizare NLP (lecțiile de NLP)
- **Naive Bayes** — clasificator baseline (lecțiile de probabilități)
- **Cross-Validation** — evaluare robustă (discutată la curs)

### Limitări
- Modelul semantic e antrenat mai mult pe engleză
- Dataset relativ mic pentru antrenare ML (700 intrări)
- Nu este aplicație medicală — **nu înlocuiește un medic!**

### Îmbunătățiri Viitoare
- [ ] Interfață Gradio/Streamlit
- [ ] API REST cu FastAPI
- [ ] Model BERT antrenat pe date medicale românești
- [ ] Căutare și după organ țintă sau frecvență

---

## ⚠️ Disclaimer

> Această aplicație este **exclusiv pentru cercetare și educație**.
> Nu înlocuiește diagnosticul medical profesional.
> Consultați întotdeauna un medic pentru probleme de sănătate.

---

## 👤 Autor

**Tibi40** — Student, Curs Machine Learning ITSchool 2026

---

<div align="center">
⭐ Dacă ți-a fost util acest proiect, lasă un star pe GitHub! ⭐
</div>
