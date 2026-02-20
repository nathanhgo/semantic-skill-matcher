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

# --- MODELAGEM DE DADOS ---
class EscoSkillGroup(Base):
    __tablename__ = 'esco_skill_groups'
    uri = Column(String(500), primary_key=True)
    termo = Column(String(500))
    parent_uri = Column(String(500))

class EscoSkill(Base):
    __tablename__ = 'esco_skills'
    id = Column(Integer, primary_key=True)
    uri = Column(String(500), unique=True)
    termo = Column(String(500))
    parent_uri = Column(String(500))
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

# --- INGESTÃO DE DADOS ---
# (A função ingest_data permanece EXATAMENTE igual à sua versão anterior, 
# não vou repeti-la inteira aqui para economizar espaço. Mantenha a sua!)
def ingest_data():
    pass 

# --- FUNÇÕES DE HIERARQUIA ---
def get_skill_hierarchy(session, start_parent_uri):
    tree = []
    current_uri = start_parent_uri
    safety_counter = 0 
    
    while current_uri and safety_counter < 6:
        group = session.query(EscoSkillGroup).filter_by(uri=current_uri).first()
        if group:
            tree.insert(0, group.termo)
            current_uri = group.parent_uri
        else:
            skill_parent = session.query(EscoSkill).filter_by(uri=current_uri).first()
            if skill_parent:
                tree.insert(0, skill_parent.termo)
                current_uri = skill_parent.parent_uri
            else:
                break
        safety_counter += 1
    return tree

def get_isco_hierarchy(session, isco_code):
    if not isco_code or len(isco_code) < 4: return []
    codes = [isco_code[:1], isco_code[:2], isco_code[:3], isco_code]
    groups = session.query(IscoGroup).filter(IscoGroup.code.in_(codes)).all()
    group_map = {g.code: g.label for g in groups}
    return [group_map.get(c, c) for c in codes]

# --- ROTA PRINCIPAL (A MÁGICA ACONTECE AQUI) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    data = {"skills": [], "occupations": []}
    texto_busca = ""
    zoom_level = request.form.get('zoom_level', 'micro') 
    
    if request.method == 'POST':
        texto_busca = request.form.get('skill_desc')
        if texto_busca:
            Session = sessionmaker(bind=engine)
            session = Session()
            vetor = model.encode(texto_busca).tolist()
            
            # --- 1. SKILLS ---
            dist_skill = EscoSkill.embedding.cosine_distance(vetor).label("dist")
            q_skills = session.query(EscoSkill, dist_skill).order_by(dist_skill).limit(6).all()
            for s, d in q_skills:
                score = (1 - d) * 100
                hierarchy = get_skill_hierarchy(session, s.parent_uri)
                
                # Regra 1: Remover o "root" inútil da ESCO
                if hierarchy and hierarchy[0].lower() in ['skills', 'knowledge', 'transversal skills and competences']:
                    hierarchy.pop(0)

                # Regra 2: Escolher o título dinâmico baseado no Zoom
                termo_exibicao = s.termo
                if zoom_level != 'micro':
                    try:
                        idx = int(zoom_level) - 1
                        if idx < len(hierarchy):
                            termo_exibicao = hierarchy[idx]
                        elif hierarchy:
                            termo_exibicao = hierarchy[-1] 
                    except ValueError:
                        pass

                data["skills"].append({
                    "termo_micro": s.termo,
                    "termo_exibicao": termo_exibicao,
                    "arvore": hierarchy, 
                    "confianca": round(score, 1),
                    "cor": "bg-success" if score > 75 else "bg-warning" if score > 50 else "bg-danger"
                })

            # --- 2. OCCUPATIONS ---
            dist_occ = EscoOccupation.embedding.cosine_distance(vetor).label("dist")
            q_occs = session.query(EscoOccupation, dist_occ).order_by(dist_occ).limit(3).all()
            for o, d in q_occs:
                score = (1 - d) * 100
                hierarchy = get_isco_hierarchy(session, o.isco_code)

                # Regra 1 (NOVA): Remover o "root" inútil da ISCO (Major Groups como "Professionals")
                if hierarchy and len(hierarchy) > 1:
                    hierarchy.pop(0)

                # Regra 2: Escolher o título dinâmico baseado no Zoom
                termo_exibicao = o.termo
                if zoom_level != 'micro':
                    try:
                        idx = int(zoom_level) - 1
                        if idx < len(hierarchy):
                            termo_exibicao = hierarchy[idx]
                        elif hierarchy:
                            termo_exibicao = hierarchy[-1]
                    except ValueError:
                        pass

                data["occupations"].append({
                    "termo_micro": o.termo,
                    "termo_exibicao": termo_exibicao,
                    "arvore": hierarchy,
                    "confianca": round(score, 1)
                })
            session.close()

    return render_template('index.html', data=data, busca_anterior=texto_busca, zoom_level=zoom_level)

if __name__ == '__main__':
    # ingest_data() # (Descomente se precisar repopular, mas seu BD já tá pronto)
    app.run(debug=True, port=5000)