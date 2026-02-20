
# Semantic Skill Matcher 🧠🔍

> **Beyond Keywords.** An intelligent, hierarchical recruitment engine that bridges the gap between unstructured CV descriptions and standardized industrial ontologies using Vector Search and JIT Translation.

### 💡 About the Project
**Semantic Skill Matcher** is a Proof of Concept (POC) designed to modernize how Recruitment & Selection platforms handle candidate data. Its mission is to solve the "vocabulary gap" in HR Tech—where a candidate writes "I fix machines" and a keyword-based system fails to match them with an "Industrial Maintenance" vacancy.

This project implements a **Semantic Search Architecture** using **PostgreSQL (pgvector)** and **Sentence-Transformers**. It maps free-text input to the **ESCO (European Skills, Competences, Qualifications and Occupations)** standard, providing not just matches, but hierarchical professional context via ISCO-08, complete with dynamic abstraction (Zoom Levels) and Just-in-Time (JIT) translation to Brazilian Portuguese.

### 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/postgresql-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)

### 🧩 The Problem vs. The Solution

#### 1. The Legacy Approach (Keyword Search)
* **The Catch:** **Rigidity**. If a recruiter searches for "Frontend Engineer", the system misses a candidate who wrote "ReactJS UI Developer". It requires exact syntax matches, leading to false negatives and lost talent.
* **The Semantic Win:** We convert text into high-dimensional vectors using a multilingual Transformer model. The system understands that "creating pipelines" is mathematically close to "Data Engineering", even if the words share no common letters.

#### 2. The Granularity Trap (Flat Lists)
* **The Catch:** ESCO has over 13,000 micro-skills. Matching a candidate to "Replace a door lock" is too granular for a recruiter looking for "Facilities Maintenance".
* **The Hierarchical Win:** **Dynamic Zoom Levels**. The engine builds a relationship tree. If the exact match is too micro, the system automatically climbs the ESCO hierarchy and groups the candidate into a broader, more useful Macro-Category (e.g., Level 1, Level 2).

#### 3. The Localization Barrier
* **The Catch:** High-quality standardized ontologies (like ESCO or O*NET) are natively in English. Presenting strict English taxonomies to Brazilian users creates a poor UX.
* **The JIT Translation Win:** The backend and vector math operate entirely in English for maximum NLP precision. However, right before rendering the UI, a **Just-in-Time translation middleware** (with an in-memory cache) translates the specific results to Portuguese, delivering the best of both worlds: global standards with local UX.

---

### 🖥️ Architecture Flow

The system operates in a pipeline designed for precision and UX:

| Stage | Component | Description |
| :--- | :--- | :--- |
| **1. Storage** | `Docker + pgvector` | A persistent volume holding the relational and vector data. |
| **2. Ingestion** | `Pandas + SQLAlchemy` | Loads ESCO datasets, maps parent-child skill relationships, and vectorizes them into 384 dimensions. |
| **3. Retrieval** | `SentenceTransformer` | Converts the user's live prompt into a vector and queries the nearest semantic neighbors (Cosine Similarity). |
| **4. Abstraction** | `Python Logic` | Climbs the ESCO/ISCO trees based on the user's selected "Zoom Level" to find the right granularity. |
| **5. Presentation** | `deep-translator` | Translates the final isolated nodes to PT-BR on-the-fly, caching results for instant subsequent loads. |

### 📊 Capability Showcase

Inputting a vague description like:
> *"I am an industrial maintenance technician. I specialize in preventive maintenance of hydraulic systems on the assembly line and operating CNC machines."*

**With Zoom Level 2 Selected, it returns (Translated to PT-BR):**
1.  **Skills:** *Manutenção de máquinas e equipamentos*, *Operação de máquinas-ferramenta*.
2.  **Occupation:** *Ajustadores e operadores de máquinas-ferramentas para trabalhar metais*.
3.  **Hierarchy:** Visually displays the tree from the Macro industry down to the micro-skill.

---

### 🚀 How to Run

1. **Clone the repository**
```bash
	   git clone [https://github.com/nathanhgo/semantic-skill-matcher.git](https://github.com/nathanhgo/semantic-skill-matcher.git)
	   cd semantic-skill-matcher
```

2.  **Start the Vector Database (Docker)** Ensure Docker is running, then spin up the PostgreSQL instance with the pgvector extension.
  ```bash
		docker compose up -d
```
    
3.  **Setup Python Environment**
    
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
  ```
    
4.  **Prepare the Ontology Data (English)** Ensure the following ESCO CSV files are in the root directory (Download from the EU ESCO Portal):
    
    -   `skills_en.csv`
        
    -   `skillGroups_en.csv`
        
    -   `broaderRelationsSkillPillar_en.csv`
        
    -   `occupations_en.csv`
        
    -   `ISCOGroups_en.csv`
        
5.  **Run the Application** _Note: In `app.py`, ensure `RESET_DB = True` for the very first run to populate the Docker volume._
    
    Bash
    
    ```
    python app.py
    
    ```
    
    Access the UI at: `http://localhost:5000`
    

----------

Developed by: @nathanhgo