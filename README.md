# Syrine Chehairi — Projects & Portfolio

Ingénieure **Logicielle & Data** (SupGalilée 2025).  
J’aime construire des solutions **propres, testées et mesurables** (Python/SQL/Docker/Tests, Data/ML & Backend).

- 🌐 **Portfolio (Streamlit)** : https://syrinechehairi-portfolio.hf.space  
- 💼 GitHub : https://github.com/syrine291100  
- 📧 Contact : syrine.chehairi@hotmail.com

---

## Table des matières
- [1. Portfolio (Streamlit)](#1-portfolio-streamlit)
- [2. Recycl-App (JavaScript)](#2-recycl-app-javascript)
- [3. Mini-jeux (Python)](#3-mini-jeux-python)
- [4. Monte Carlo — Simulation d’incertitudes](#4-monte-carlo--simulation-dincertitudes)
- [5. PlantAI — Classification de plantes](#5-plantai--classification-de-plantes)
- [6. Licences](#6-licences)

---

## 1) Portfolio (Streamlit)

**Repo** : https://github.com/syrine291100/Portfolio  
**Démo** : https://syrinechehairi-portfolio.hf.space

Portfolio **Streamlit** multipage (pastel, responsive) pour présenter compétences, projets et expériences.

### ✨ Fonctionnalités
- Barre de navigation **sticky** (pastel)
- Pages : Accueil, Compétences, Projets (descriptions), Expériences, Formations, Contact
- Cartes + tags, design sobre

### 🧰 Stack
Python 3.10+, Streamlit, Pillow

### ▶️ Lancer en local
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### ☁️ Déploiement
- Streamlit Cloud (New app → `app.py`)
- Hugging Face Spaces (SDK: streamlit, `app_file: app.py`)

---

## 2) Recycl-App (JavaScript)

**Repo** : https://github.com/syrine291100/Recycl-App

Application web simple pour **aider au tri sélectif** (catégories, conseils, recherche).

### ✨ Fonctionnalités
- Recherche d’objet/matériau (ex. “bouteille”)
- Catégorie cible (verre, papier, ordures, déchèterie…)
- Bonnes pratiques (rincer, plier, enlever bouchon)
- UI légère et responsive

### 🧰 Stack
HTML • CSS • JavaScript (vanilla)  
*(option) `data/items.json` pour la base de correspondances*

### ▶️ Lancer
- Ouvrir `index.html` directement  
ou
```bash
# serveur local
python -m http.server 5500
# puis http://localhost:5500
```

### 📁 Structure suggérée
```
Recycl-App/
  index.html
  css/
  js/
    app.js
  data/
    items.json
```

---

## 3) Mini-jeux (Python)

**Repo** : https://github.com/syrine291100/mini-jeux

Collection de **mini-jeux en Python** (logique, algo, I/O ; certains avec graphisme léger).  
Un **menu `main.py`** permet de tout lancer.

### 🔧 Prérequis
- Python 3.10+
- (si un jeu utilise une lib externe) `pip install <lib>` — ex. `pygame`

### ▶️ Lancer avec le menu
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python main.py
```

### ▶️ Lancer un jeu directement
```bash
python snake.py
python "tic-tac-toe.py"   # guillemets recommandés sur Windows
```

### 🎮 Jeux inclus

| Jeu | Fichier |
|---|---|
| 2048 | `game_2048.py` |
| Breakout | `breackout.py` *(orthographe actuelle du fichier)* |
| Flappy | `flappy.py` |
| Devine le nombre | `guess_number.py` |
| High scores (utilitaire) | `high_scores.py` |
| Quiz de math | `math_quiz.py` |
| Memory | `memory.py` |
| Pendu | `pendu.py` |
| Pong | `pong.py` |
| Réaction (timer) | `reaction_timer.py` |
| Simon Says | `simon_says.py` |
| Taquin / Sliding puzzle | `slidding_puzzle.py` |
| Snake | `snake.py` |
| Sudoku | `sudoku.py` |
| Tic-Tac-Toe (morpion) | `tic-tac-toe.py` |
| Whack-a-Mole | `whack_a_mole.py` |

> Dossier `test/` : utilitaires/tests éventuels.

### ✅ TODO
- Renommer plus tard `breackout.py` → `breakout.py` et `slidding_puzzle.py` → `sliding_puzzle.py`  
- Ajouter `requirements.txt` si certains jeux utilisent `pygame`/autres  
- Centraliser les **scores** (`high_scores.py`) + petits **tests** (pytest)

---

## 4) Monte Carlo — Simulation d’incertitudes

**Repo (recommandé)** : https://github.com/syrine291100/monte-carlo-uncertainty *(même si le code n’est pas publiable)*

> Contexte industriel : le **code métier n’est pas publié**. Le dépôt sert de **documentation technique** + **exemple générique**.

### 🎯 Objectif
Estimer l’**incertitude** d’un indicateur via **échantillonnage Monte Carlo** :
- Entrées aléatoires (lois normales, uniformes…)
- Fonction métier potentiellement coûteuse
- Sorties : moyenne, variance, quantiles/IC

### 🧠 Approche
- Génération pseudo-aléatoire **vectorisée** (NumPy)
- Simulation **par lots** (limite mémoire)
- Statistiques **en ligne** (streaming) pour éviter de stocker chaque tirage
- (Option) multi-thread / GPU

#### Pseudocode
```python
N = 1_000_000
stats = OnlineStats()
for X in batched_samples(N, batch_size=10_000):
    Y = f_metier(X)
    stats.update(Y)
report = stats.summary()
```

### 🔬 Qualité / V&V
Tests unitaires (pytest) : seed/reproductibilité, bornes, distributions ; non-régression ; traçabilité (versions, paramètres, seeds).

### 📦 Arborescence conseillée
```
monte-carlo-uncertainty/
├─ README.md
├─ examples/
│  └─ toy_example.ipynb     # démo générique non confidentielle
├─ requirements.txt         # numpy, matplotlib, (scipy), jupyter
└─ LICENSE
```

---

## 5) PlantAI — Classification de plantes

**Repo** : https://github.com/syrine291100/plantAIbdd *(peut contenir notes/code partiel ; données privées)*

### 🧠 Approche
- Transfert d’apprentissage (ResNet/MobileNet)
- Prétraitements : resize, normalisation, augmentations
- Split stratifié train/val/test
- Métriques : Accuracy, F1, confusion matrix
- Export modèle + script d’inférence

### 📁 Structure suggérée
```
plantAI/
  data/                 # non versionnée
    train/ val/ test/
  src/
    train.py
    dataset.py
    model.py
    infer.py
  notebooks/
  requirements.txt
  README.md
```

### ▶️ Entraînement (template)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data ./data --epochs 20 --batch-size 32 --model mobilenet_v3 --img-size 224
```

### 🔎 Inférence (template)
```bash
python src/infer.py --weights runs/best.pt --image path/to/leaf.jpg
```

### ✅ TODO
Notebook d’exemple (dataset public), rapports d’évaluation (confusion matrix), script d’éval croisée.

---

## 6) Licences
Sauf mention contraire dans chaque repo, licence **MIT**.
