import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
# --- NOVA IMPORTAÇÃO ---
from deep_translator import GoogleTranslator

app = Flask(__name__)

# --- CONFIGURAÇÃO ---
DATABASE_URL = "postgresql://usuario:senha@localhost:5433/rh_poc"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
EMBEDDING_DIM = 384
RESET_DB = True

print("--- CARREGANDO CÉREBRO DA IA ---")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

# --- SISTEMA DE TRADUÇÃO SOB DEMANDA (POC) ---
print("--- INICIANDO MOTOR DE TRADUÇÃO (PT-BR) ---")
translator = GoogleTranslator(source='en', target='pt')
CACHE_TRADUCAO = {}

def traduzir_ptbr(texto):
    """Traduz o texto para PT-BR e salva em cache para buscas futuras ficarem instantâneas."""
    if not texto: 
        return texto
    if texto in CACHE_TRADUCAO:
        return CACHE_TRADUCAO[texto]
    
    try:
        # Tenta traduzir
        traduzido = translator.translate(texto)
        CACHE_TRADUCAO[texto] = traduzido
        return traduzido
    except Exception as e:
        # Se a internet cair ou o Google bloquear, retorna em inglês (Fallback seguro)
        print(f"Erro ao traduzir '{texto}': {e}")
        return texto

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
def ingest_data():
    if RESET_DB:
        print("!!! LIMPANDO BANCO DE DADOS PARA NOVA ESTRUTURA !!!")
        Base.metadata.drop_all(engine)
    
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    # 1. Carregar Hierarquia ISCO
    if session.query(IscoGroup).count() == 0:
        print("Ingerindo ISCO Groups...")
        try:
            df = pd.read_csv('ISCOGroups_en.csv')
            df['code'] = df['code'].astype(str)
            df = df.drop_duplicates(subset=['code'], keep='last')
            for _, row in df.iterrows():
                session.add(IscoGroup(code=str(row['code']), label=row['preferredLabel']))
            session.commit()
        except Exception as e: print(f"Erro ISCO: {e}")

    # 2. Carregar Mapa de Relações de Skills (Pai e Filho)
    print("Carregando Mapa de Relações...")
    rel_dict = {}
    try:
        df_rel = pd.read_csv('broaderRelationsSkillPillar_en.csv', usecols=['conceptUri', 'broaderUri'])
        df_rel = df_rel.drop_duplicates(subset=['conceptUri'], keep='first')
        rel_dict = pd.Series(df_rel.broaderUri.values, index=df_rel.conceptUri).to_dict()
    except Exception as e:
        print(f"Aviso: Mapa de relações não carregado ({e}). Hierarquia de skills ficará vazia.")

    # 3. Carregar Grupos de Skills (Níveis Macro)
    if session.query(EscoSkillGroup).count() == 0:
        print("Ingerindo Skill Groups...")
        try:
            df_grp = pd.read_csv('skillGroups_en.csv', usecols=['conceptUri', 'preferredLabel'])
            objs = []
            for _, row in df_grp.iterrows():
                uri = row['conceptUri']
                parent = rel_dict.get(uri) 
                objs.append(EscoSkillGroup(uri=uri, termo=row['preferredLabel'], parent_uri=parent))
            session.add_all(objs)
            session.commit()
        except Exception as e: print(f"Erro Skill Groups: {e}")

    # 4. Carregar Skills (Com vetorização)
    if session.query(EscoSkill).count() == 0:
        print("Ingerindo Skills...")
        try:
            df_en = pd.read_csv('skills_en.csv', usecols=['conceptUri', 'preferredLabel'])
            df_en = df_en.drop_duplicates(subset=['preferredLabel'])
            
            termos = df_en['preferredLabel'].tolist()
            uris = df_en['conceptUri'].tolist()
            
            for i in range(0, len(termos), 64):
                batch_termos = termos[i:i+64]
                batch_uris = uris[i:i+64]
                vetores = model.encode(batch_termos)
                
                objs = []
                for termo, uri, vetor in zip(batch_termos, batch_uris, vetores):
                    parent = rel_dict.get(uri) 
                    objs.append(EscoSkill(uri=uri, termo=termo, parent_uri=parent, embedding=vetor.tolist()))
                session.add_all(objs)
                session.commit()
                print(f"Skills: {i}/{len(termos)}", end='\r')
        except Exception as e: print(f"Erro Skills: {e}")

    # 5. Carregar Occupations
    if session.query(EscoOccupation).count() == 0:
        print("\nIngerindo Occupations...")
        try:
            df_occ = pd.read_csv('occupations_en.csv', usecols=['preferredLabel', 'iscoGroup'])
            df_occ = df_occ.drop_duplicates(subset=['preferredLabel'])
            termos = df_occ['preferredLabel'].dropna().tolist()
            termo_to_isco = pd.Series(df_occ.iscoGroup.values, index=df_occ.preferredLabel).to_dict()
            
            for i in range(0, len(termos), 64):
                batch = termos[i:i+64]
                vetores = model.encode(batch)
                objs = []
                for t, v in zip(batch, vetores):
                    code = str(termo_to_isco.get(t, "0000"))
                    if code.lower() == 'nan': code = "0000"
                    objs.append(EscoOccupation(termo=t, isco_code=code, embedding=v.tolist()))
                session.add_all(objs)
                session.commit()
                print(f"Occs: {i}/{len(termos)}", end='\r')
        except Exception as e: print(f"Erro Occs: {e}")

    session.close()

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

# --- ROTA PRINCIPAL ---
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
                
                if hierarchy and hierarchy[0].lower() in ['skills', 'knowledge', 'transversal skills and competences']:
                    hierarchy.pop(0)

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

                # APLICANDO A TRADUÇÃO ANTES DE ENVIAR PARA A TELA
                data["skills"].append({
                    "termo_micro": traduzir_ptbr(s.termo),
                    "termo_exibicao": traduzir_ptbr(termo_exibicao),
                    "arvore": [traduzir_ptbr(node) for node in hierarchy], 
                    "confianca": round(score, 1),
                    "cor": "bg-success" if score > 75 else "bg-warning" if score > 50 else "bg-danger"
                })

            # --- 2. OCCUPATIONS ---
            dist_occ = EscoOccupation.embedding.cosine_distance(vetor).label("dist")
            q_occs = session.query(EscoOccupation, dist_occ).order_by(dist_occ).limit(3).all()
            for o, d in q_occs:
                score = (1 - d) * 100
                hierarchy = get_isco_hierarchy(session, o.isco_code)

                if hierarchy and len(hierarchy) > 1:
                    hierarchy.pop(0)

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

                # APLICANDO A TRADUÇÃO ANTES DE ENVIAR PARA A TELA
                data["occupations"].append({
                    "termo_micro": traduzir_ptbr(o.termo),
                    "termo_exibicao": traduzir_ptbr(termo_exibicao),
                    "arvore": [traduzir_ptbr(node) for node in hierarchy],
                    "confianca": round(score, 1)
                })
            session.close()

    return render_template('index.html', data=data, busca_anterior=texto_busca, zoom_level=zoom_level)

if __name__ == '__main__':
    ingest_data() 
    app.run(debug=True, port=5000)