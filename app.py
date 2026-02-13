import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- CONFIGURAÇÃO ---
DATABASE_URL = "postgresql://usuario:senha@localhost:5433/rh_poc"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
EMBEDDING_DIM = 384

RESET_DB = False 

print("--- CARREGANDO CÉREBRO DA IA ---")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

# --- Modelagem ---
class EscoSkill(Base):
    __tablename__ = 'esco_skills'
    id = Column(Integer, primary_key=True)
    termo = Column(String(500), unique=True)
    embedding = Column(Vector(EMBEDDING_DIM))

class EscoOccupation(Base):
    __tablename__ = 'esco_occupations'
    id = Column(Integer, primary_key=True)
    termo = Column(String(500), unique=True)
    isco_code = Column(String(10)) 
    embedding = Column(Vector(EMBEDDING_DIM))

class IscoGroup(Base):
    __tablename__ = 'isco_groups'
    code = Column(String(10), primary_key=True)
    label = Column(String(500))

# --- Patch ISCO em Inglês ---
def fix_isco_top_levels(session):
    """Garante os rótulos corretos em Inglês para os grupos principais"""
    correcoes = {
        "0": "Armed forces occupations",
        "1": "Managers",
        "2": "Professionals",
        "3": "Technicians and associate professionals",
        "4": "Clerical support workers",
        "5": "Service and sales workers",
        "6": "Skilled agricultural, forestry and fishery workers",
        "7": "Craft and related trades workers",
        "8": "Plant and machine operators and assemblers",
        "9": "Elementary occupations"
    }
    
    print("Aplicando patch ISCO (English)...")
    for code, label in correcoes.items():
        grupo = session.query(IscoGroup).get(code)
        if grupo:
            grupo.label = label
        else:
            session.add(IscoGroup(code=code, label=label))
    session.commit()

# --- Ingestão ---
def ingest_data():
    # 1. Reset Total (Se solicitado)
    if RESET_DB:
        print("!!! ALERTA: DESTRUINDO BANCO DE DADOS ANTIGO PARA MIGRAÇÃO !!!")
        Base.metadata.drop_all(engine)
    
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    # Aplica Correção Hierárquica
    fix_isco_top_levels(session)

    # Ingestão Skills
    if session.query(EscoSkill).count() == 0:
        print("Ingerindo Skills (English)...")
        try:
            df = pd.read_csv('skills_en.csv')
            termos = df['preferredLabel'].dropna().unique().tolist()
            
            for i in range(0, len(termos), 64):
                batch = termos[i:i+64]
                vetores = model.encode(batch)
                objs = [EscoSkill(termo=t, embedding=v.tolist()) for t, v in zip(batch, vetores)]
                session.add_all(objs)
                session.commit()
                print(f"Skills: {i}/{len(termos)}", end='\r')
        except Exception as e: 
            print(f"Erro Skills: {e}")
            session.rollback()

    # Ingestão ISCO
    if session.query(IscoGroup).count() < 15:
        print("\nIngerindo ISCO Groups (English)...")
        try:
            df = pd.read_csv('ISCOGroups_en.csv')
            df['code'] = df['code'].astype(str)
            df = df.drop_duplicates(subset=['code'], keep='last')
            
            for _, row in df.iterrows():
                if not session.query(IscoGroup).get(str(row['code'])):
                    session.add(IscoGroup(code=str(row['code']), label=row['preferredLabel']))
            session.commit()
        except Exception as e: 
            print(f"Erro ISCO: {e}")
            session.rollback()

    # Ingestão Occupations
    if session.query(EscoOccupation).count() == 0:
        print("\nIngerindo Occupations (English)...")
        try:
            df = pd.read_csv('occupations_en.csv', usecols=['preferredLabel', 'iscoGroup'])
            df = df.drop_duplicates(subset=['preferredLabel'])
            termos = df['preferredLabel'].dropna().tolist()
            termo_to_isco = pd.Series(df.iscoGroup.values, index=df.preferredLabel).to_dict()
            
            for i in range(0, len(termos), 64):
                batch = termos[i:i+64]
                vetores = model.encode(batch)
                objs = []
                for t, v in zip(batch, vetores):
                    code = str(termo_to_isco.get(t, "0000"))
                    if str(code).lower() == 'nan': code = "0000"
                    objs.append(EscoOccupation(termo=t, isco_code=code, embedding=v.tolist()))
                session.add_all(objs)
                session.commit()
                print(f"Occs: {i}/{len(termos)}", end='\r')
        except Exception as e:
            print(f"Erro Occs: {e}")
            session.rollback()
    
    session.close()

# --- Funções Auxiliares ---
def get_isco_hierarchy(session, isco_code):
    if not isco_code or len(isco_code) < 4: return []
    codes = [isco_code[:1], isco_code[:2], isco_code[:3], isco_code]
    groups = session.query(IscoGroup).filter(IscoGroup.code.in_(codes)).all()
    group_map = {g.code: g.label for g in groups}
    
    hierarchy = []
    for c in codes:
        hierarchy.append({"code": c, "label": group_map.get(c, c)})
    return hierarchy

# --- Rota ---
@app.route('/', methods=['GET', 'POST'])
def index():
    data = {"skills": [], "occupations": []}
    texto_busca = ""
    
    if request.method == 'POST':
        texto_busca = request.form.get('skill_desc')
        if texto_busca:
            Session = sessionmaker(bind=engine)
            session = Session()
            vetor = model.encode(texto_busca).tolist()
            
            # Skills
            dist_skill = EscoSkill.embedding.cosine_distance(vetor).label("dist")
            q_skills = session.query(EscoSkill, dist_skill).order_by(dist_skill).limit(6).all()
            for s, d in q_skills:
                score = (1 - d) * 100
                data["skills"].append({
                    "termo": s.termo,
                    "confianca": round(score, 1),
                    "cor": "bg-success" if score > 75 else "bg-warning" if score > 50 else "bg-danger"
                })

            # Occupations
            dist_occ = EscoOccupation.embedding.cosine_distance(vetor).label("dist")
            q_occs = session.query(EscoOccupation, dist_occ).order_by(dist_occ).limit(3).all()
            for o, d in q_occs:
                score = (1 - d) * 100
                data["occupations"].append({
                    "termo": o.termo,
                    "confianca": round(score, 1),
                    "arvore": get_isco_hierarchy(session, o.isco_code)
                })
            session.close()

    return render_template('index.html', data=data, busca_anterior=texto_busca)

if __name__ == '__main__':
    ingest_data()
    app.run(debug=True, port=5000)