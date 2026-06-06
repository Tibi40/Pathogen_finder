# 🦠 Finder Agenți Patogeni — ML-Powered Pathogen Search System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-AI-orange?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=for-the-badge&logo=googlecolab)

**Un sistem AI care primește numele unei boli și identifică instant agenții patogeni asociați, tipul lor, frecvențele și organele afectate.**

[▶️ Deschide în Google Colab](https://colab.research.google.com/github/Tibi40/pathogen-finder/blob/main/notebook.ipynb)

</div>

---

## 📋 Cuprins

1. [Despre Proiect](#despre-proiect)
2. [Cum acționează virusii și paraziții în corp](#cum-actioneaza)
3. [Cum ajută AI-ul să îi găsească](#cum-ajuta-ai)
4. [Arhitectura Sistemului](#arhitectura)
5. [Dataset](#dataset)
6. [Modele AI](#modele-ai)
7. [Rezultate și Comparație](#rezultate)
8. [Cum se folosește](#cum-se-foloseste)
9. [Structura Proiectului](#structura)
10. [Tehnologii](#tehnologii)

---

## 🎯 Despre Proiect

### Problema
Într-un catalog medical cu **peste 700 de agenți patogeni**, identificarea manuală a celor asociați cu o anumită boală durează minute sau ore.

### Soluția
Un sistem AI care primește **un singur cuvânt** (numele bolii) și returnează în **sub 1 secundă** lista agenților patogeni, tipul, frecvența Hz și organele afectate — toate validate automat cu **Pydantic**.

### Contextul de utilizare
> *"Lucrez într-un laborator de cercetare și am nevoie să identific rapid ce agenți patogeni sunt asociați cu o boală specifică, fără să parcurg manual sute de înregistrări."*

---

## 🦠 Cum Acționează Virusii și Paraziții în Corp

### Schema Generală de Infecție

```
AGENT PATOGEN (Virus / Bacterie / Parazit)
          │
          ▼
    ┌─────────────────────────────────────┐
    │         POARTA DE INTRARE           │
    │  Piele │ Respirator │ Digestiv │ Sânge│
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │         PRIMA LINIE DE APĂRARE      │
    │    Sistem imunitar înnăscut         │
    │  (Neutrofile, Macrofage, NK cells)  │
    └─────────────────────────────────────┘
          │
     ┌────┴────┐
     │         │
  SUCCES    EȘEC → Agentul trece mai departe
     │                    │
  Corp sănătos            ▼
              ┌─────────────────────────┐
              │   RĂSPÂNDIRE ÎN ORGANISM │
              │  Via sânge (hematogen)   │
              │  Via limfă (limfogen)    │
              │  Via nervi (neurotropic) │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │    ORGANE ȚINTĂ         │
              │ Ficat │ Plămâni │ Creier │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │     MANIFESTARE BOALĂ   │
              │ Simptome locale+sistemice│
              └─────────────────────────┘
```

### 🔵 Cum Acționează VIRUSII

| Etapă | Ce se întâmplă | Exemplu |
|-------|---------------|---------|
| **1. Atașare** | Virusul se lipește de receptori specifici ai celulei | Coronavirusul pe receptorii ACE2 |
| **2. Penetrare** | Virusul intră în celulă și injectează ADN/ARN viral | Material genetic preluat |
| **3. Replicare** | Folosește "utilajele" celulei pentru a se copia | Produce mii de copii |
| **4. Eliberare** | Virusurile noi sparg celula și infectează altele | Celula moare |
| **5. Răspuns imun** | Corpul atacă celulele infectate | Febră, inflamație |

```
CELULĂ SĂNĂTOASĂ     →    CELULĂ INFECTATĂ     →    CELULE NOI INFECTATE
    [Receptori]           [Virus se atașează]        [Replicare masivă]
  Funcționează         Virusul preia controlul      Celula explodează
   normal               Produce copii virale         Virusuri libere
```

### 🟠 Cum Acționează PARAZIȚII

| Tip Parazit | Mecanism | Organe Afectate | Exemplu din Dataset |
|-------------|----------|-----------------|---------------------|
| **Protozoare** | Intră în celule, se multiplică | Sânge, ficat, creier | Toxoplasma, Entamoeba |
| **Helminți** | Migrează prin țesuturi, fură nutrienți | Intestin, ficat, plămâni | Taenia, Echinococcus |
| **Ectoparaziți** | Trăiesc pe suprafața corpului | Piele | Diverse specii |

```
PARAZIT INTRAT ÎN CORP
        │
        ▼
   Traversează barierele (piele, mucoase)
        │
        ▼
   ┌────────────────────────────────┐
   │    STRATEGII DE SUPRAVIEȚUIRE  │
   │  Ascundere de sistemul imun    │
   │  Mimează proteine proprii      │
   │  Suprimă răspunsul imun        │
   └────────────────────────────────┘
        │
        ▼
   Colonizează organ țintă
        ├── Fură nutrienți din gazdă
        ├── Produce toxine
        ├── Distruge țesut local
        └── Declanșează reacție inflamatorie
```

### 🔴 Cum Acționează BACTERIILE

```
BACTERIE
    │
    ├── Produce TOXINE
    │       ├── Exotoxine (secretate direct în corp)
    │       └── Endotoxine (eliberate când bacteria moare)
    │
    ├── INVAZIA DIRECTĂ a țesuturilor
    │
    ├── Evadare din sistemul imun
    │       ├── Capsulă protectoare (ex: Pneumococ)
    │       └── Biofilm (bacterii care trăiesc în grup)
    │
    └── SIMPTOME
            ├── Locale: inflamație, puroi, roșeață
            └── Sistemice: febră, frisoane, sepsis
```

---

## 🤖 Cum Ajută AI-ul să Îi Găsească

### Problema Clasică de Căutare (Fără AI)

```
Utilizator: "Vreau să știu despre ACNEE"
              │
              ▼
   Parcurge manual 700+ rânduri din catalog
              │
              ▼
   Caută cuvântul "ACNEE" în fiecare rând
              │
              ▼
   Timp: 5-10 minute
   Risc: ratează "acne", "Acne vulgaris", "acneea"
```

### Cu AI — Fluxul Nostru

```
Utilizator scrie: "acnee"
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                     SISTEMUL AI                          │
│                                                          │
│  STRAT 1: Căutare Directă                               │
│  ┌──────────────────────────────────────────────┐       │
│  │ Caută exact "acnee" în baza de date          │       │
│  │ Rezultat: Propionibacterium acnes, Dioxina   │       │
│  │ Timp: mai puțin de 10ms                      │       │
│  └──────────────────────────────────────────────┘       │
│           │                                              │
│           │ (dacă nu găsește destule rezultate)          │
│           ▼                                              │
│  STRAT 2: TF-IDF (mai inteligent)                       │
│  ┌──────────────────────────────────────────────┐       │
│  │ Caută cuvinte similare și parțiale           │       │
│  │ Înțelege că "acn" este parte din "acnee"    │       │
│  │ Timp: mai puțin de 50ms                      │       │
│  └──────────────────────────────────────────────┘       │
│           │                                              │
│           │ (dacă tot nu e suficient)                    │
│           ▼                                              │
│  STRAT 3: Semantic AI (cel mai avansat)                 │
│  ┌──────────────────────────────────────────────┐       │
│  │ Înțelege SENSUL textului                     │       │
│  │ "coșuri pe față" poate găsi "acnee"         │       │
│  │ Timp: mai puțin de 500ms                     │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  VALIDARE PYDANTIC                        │
│  Verifică că fiecare rezultat are:                       │
│  ✅ Nume agent patogen (string, obligatoriu)              │
│  ✅ Tip (Bacterie/Parazit/Virus)                          │
│  ✅ Frecvență Hz                                          │
│  ✅ Lista boli asociate (list of strings)                 │
│  ✅ Organe afectate (list of strings)                     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   Rezultat structurat afișat utilizatorului
   Timp total: mai puțin de 1 secundă
```

### De ce Pydantic?

```python
# FARA Pydantic — risc de date gresite:
rezultat = {"nume": "Toxoplasma", "frecventa": None}
# None poate crapa aplicatia!

# CU Pydantic — validare automata:
agent = AgentPatogen(
    nume="Toxoplasma",
    tip="Parazit",
    frecventa="395,000",
    boli_asociate=["Toxoplasmoza", "Toxoplasmoza cerebrala"],
    organe_tinta=["blood", "cerebrum", "hepar"],
    numar_boli=2
)
# Daca un camp lipseste → eroare clara, nu date corupte
# Exportabil automat in JSON
```

---

## 🏗️ Arhitectura Sistemului

```
┌──────────────────────────────────────────────────────────────┐
│                       UTILIZATOR                              │
│                 Scrie numele bolii în chat                     │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   INTERFAȚA DE INPUT                          │
│             BOALA_CAUTATA = "numele bolii"                    │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  PREPROCESARE TEXT                            │
│   Lowercase + Eliminare caractere + Normalizare spații        │
└──────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   MODEL 1    │  │   MODEL 2    │  │   MODEL 3    │
   │  Căutare     │  │   TF-IDF     │  │  Semantic    │
   │  Directă     │  │  Vectorizer  │  │  AI (BERT)   │
   │  < 10ms      │  │  < 50ms      │  │  < 500ms     │
   └──────────────┘  └──────────────┘  └──────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  MOTOR DE DECIZIE                             │
│   Dacă Model 1 gaseste ≥ 2 rezultate → returneaza            │
│   Altfel incearca Model 2                                     │
│   Altfel foloseste Model 3                                    │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  VALIDARE PYDANTIC                            │
│   AgentPatogen: nume, tip, frecventa, boli, organe           │
│   RezultatCautare: boala, agenti, numar, mesaj               │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   OUTPUT STRUCTURAT                           │
│   Nume agent │ Tip │ Frecventa Hz │ Organe │ Boli asociate   │
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
| `pathogen_type` | categorie | Bacterie / Parazit / Virus | Parazit |
| `frequency_hz` | numeric | Frecvența de rezonanță în Hz | 395,000 |
| `disease_count` | numeric | Numărul de boli asociate | 2 |
| `associated_diseases` | text | Lista bolilor (separator ;) | Toxoplasmoza; Toxoplasmoza cerebrală |
| `target_organs` | text | Organe afectate (separator ;) | blood; cerebrum totalis; hepar |

### Distribuția Tipurilor de Agenți Patogeni

```
Bacterie     ████████████████████  35%
Parazit      ███████████████       25%
Necunoscut   ████████████          20%
Coduri num.  ████████              15%
Altele       █████                  5%
```

### Top 10 Organe cel mai des Afectate

```
blood (sânge)        ████████████████████████  420 agenți
hepar (ficat)        █████████████████████     380 agenți
pulma (plămâni)      ████████████████████      350 agenți
cerebrum totalis     ███████████████           280 agenți
cutis (piele)        ██████████████            260 agenți
lymph (limfă)        █████████████             240 agenți
os (oase)            ████████████              220 agenți
myocardium (inimă)   ███████████               200 agenți
medulla spinalis     █████████                 160 agenți
erythrocutes         ████████                  140 agenți
```

---

## 🧠 Modele AI

### Model 1: Căutare Directă

**Pro:** Extrem de rapid, precis pentru termeni exacți în română
**Contra:** Nu găsește sinonime sau variații ortografice

### Model 2: TF-IDF

**Formula folosită:**
```
              A · B
sim(A,B) = ─────────    (Cosinus Similarity)
            |A| × |B|

Rezultat: 0 = total diferit, 1 = identic
```

**Pro:** Găsește potriviri parțiale, ignoră cuvintele comune
**Contra:** Nu înțelege sensul, doar frecvența cuvintelor

### Model 3: Semantic AI (Sentence Transformers)

**Modelul:** `all-MiniLM-L6-v2` (BERT-based)
**Embedding:** Vector de 384 dimensiuni per text

```
"boală de piele cu coșuri"
        │
        ▼ (BERT encoding)
[0.23, -0.11, 0.67, 0.02, ...]  ← 384 numere = sensul textului
        │
        ▼
Comparație cu toți agenții patogeni din baza de date
        │
        ▼
Returnează cei mai apropiați semantic
```

**Pro:** Înțelege sensul, funcționează cu limbaj natural
**Contra:** Mai lent, antrenat mai mult pe engleză

---

## 📈 Rezultate și Comparație Modele

| Criteriu | Căutare Directă | TF-IDF | Semantic AI |
|----------|----------------|--------|-------------|
| **Viteză** | mai puțin de 10ms | mai puțin de 50ms | mai puțin de 500ms |
| **Precizie termeni exacți** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Înțelege sinonime** | Nu | Parțial | Da |
| **Funcționează offline** | Da | Da | Da (model mic) |
| **Recomandat pentru** | Termeni exacți | Termeni parțiali | Limbaj natural |

### Scoruri pe Boli de Test

| Boală Căutată | Căutare Directă | TF-IDF | Semantic |
|---------------|----------------|--------|---------|
| ACNEE | 4 agenți | 4 agenți | 5 agenți |
| sifilis | 3 agenți | 3 agenți | 5 agenți |
| toxoplasmoza | 2 agenți | 2 agenți | 5 agenți |
| tenie | 5 agenți | 5 agenți | 5 agenți |
| tetanos | 4 agenți | 4 agenți | 5 agenți |

**Modelul final ales:** Căutare Directă + fallback TF-IDF + Semantic

---

## 🚀 Cum se Folosește

### Deschide în Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tibi40/pathogen-finder/blob/main/notebook.ipynb)

**Pasul 1:** Click pe butonul de mai sus

**Pasul 2:** Runtime → Run All (așteaptă 3-5 minute)

**Pasul 3:** Mergi la **Pasul 12** din notebook și schimbă:
```python
BOALA_CAUTATA = "scrie boala ta aici"
```

**Pasul 4:** Apasă Shift+Enter și vezi rezultatele!

### Exemple de Căutări

```python
BOALA_CAUTATA = "ACNEE"           # boli de piele
BOALA_CAUTATA = "sifilis"         # boli infecțioase
BOALA_CAUTATA = "toxoplasmoza"    # boli parazitare
BOALA_CAUTATA = "tenie"           # paraziți intestinali
BOALA_CAUTATA = "TETANOS"         # boli bacteriene
BOALA_CAUTATA = "boala Lyme"      # boli transmise de căpușe
BOALA_CAUTATA = "TUSE"            # boli respiratorii
```

### Exemplu Output

```
🔍 Am găsit 4 agenți patogeni asociați cu "ACNEE"
════════════════════════════════════════════════════

1. 🦠 Propionibacterium acnes
   Tip: Bacterie
   Frecvența: 383,750 Hz
   Număr boli asociate: 7
   Boli principale: Infecții | Endocardita bacteriană | ACNEE
   Organe afectate: cutis | os | periosteum

2. 🦠 Dioxina
   Tip: Necunoscut
   Frecvența: cancerogen Hz
   Număr boli asociate: 10
   Boli principale: ACNEE | LEUCEMIE | LIMFOM
   Organe afectate: cutis | granulocytes | os
```

---

## 📁 Structura Proiectului

```
pathogen-finder/
│
├── notebook.ipynb              ← Codul principal
├── README.md                   ← Această documentație
├── requirements.txt            ← Librăriile necesare
│
├── data/
│   ├── vega_pathogens_catalog.csv   ← Dataset original
│   └── patogeni_procesati.csv       ← Dataset procesat
│
├── models/
│   ├── model_tfidf.pkl              ← Model TF-IDF salvat
│   └── embeddings.npy               ← Vectori AI salvați
│
└── results/
    ├── grafic1_tipuri.png
    ├── grafic2_boli_per_agent.png
    ├── grafic3_top_agenti.png
    ├── grafic4_organe.png
    ├── grafic5_proportii.png
    └── grafic6_comparatie.png
```

---

## 🛠️ Tehnologii Folosite

| Tehnologie | Utilizare |
|-----------|-----------|
| **Python 3.10+** | Limbajul principal |
| **Pydantic v2** | Validarea și structurarea datelor |
| **Sentence Transformers** | Modelul AI semantic (BERT) |
| **Scikit-learn** | TF-IDF Vectorizer + Cosinus Similarity |
| **Pandas** | Procesarea datelor CSV |
| **Matplotlib + Seaborn** | Generarea graficelor EDA |
| **NumPy** | Calcule matematice vectoriale |
| **Google Colab** | Mediul de rulare cu GPU gratuit |

---

## 🔬 Concluzii

**Ce am realizat:**
- ✅ Sistem funcțional de căutare a agenților patogeni după boală
- ✅ 3 modele AI comparate și evaluate
- ✅ Validare automată a datelor cu **Pydantic**
- ✅ 6 vizualizări EDA ale dataset-ului
- ✅ Output structurat în format JSON

**Ce am învățat:**
Combinarea mai multor modele de căutare dă rezultate mai bune decât un singur model. Pydantic este esențial când lucrezi cu date medicale — nu poți permite câmpuri lipsă sau tipuri greșite.

**Limitări:**
- Modelul semantic e antrenat pe engleză, nu pe română
- Nu este o aplicație medicală — nu înlocuiește un medic

**Îmbunătățiri Viitoare:**
- Interfață web cu Gradio sau Streamlit
- API REST cu FastAPI
- Căutare și după organe țintă

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
