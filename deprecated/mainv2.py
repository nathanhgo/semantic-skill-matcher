import pandas as pd
import csv
from typing import List, Tuple
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

# --- Configuração ---
print("Carregando modelo de IA...")
# Dica: Use 'cpu' explicitamente se não tiver GPU configurada corretamente
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
EMBEDDING_DIM = 384 

DATABASE_URL = "postgresql://usuario:senha@localhost:5433/rh_poc"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# --- Modelagem ---
class EscoSkill(Base):
    __tablename__ = 'esco_skills'
    id = Column(Integer, primary_key=True)
    termo = Column(String, unique=True)
    # Recomendado: Adicionar índice HNSW para produção (performance)
    embedding = Column(Vector(EMBEDDING_DIM)) 

    def __repr__(self):
        return f"<Skill: {self.termo}>"

# --- Funções de Apoio ---

def setup_database():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def get_embedding(text: str) -> List[float]:
    return model.encode(text).tolist()

def seed_mock_data(session):
    """Dados falsos para teste rápido"""
    print("Populando banco com dados MOCK...")
    skills_mock = [
        "Operador de Empilhadeira", "Soldagem MIG/MAG", 
        "Controle de Qualidade", "Usinagem CNC", 
        "Gestão de Equipe", "Desenvolvimento Python"
    ]
    objetos = [EscoSkill(termo=s, embedding=get_embedding(s)) for s in skills_mock]
    session.add_all(objetos)
    session.commit()

def ingest_real_esco_csv(session, file_path='skills_pt.csv'):
    """
    Como usar na vida real:
    1. Baixe o CSV da ESCO (language=pt)
    2. O CSV geralmente tem colunas: 'conceptUri', 'preferredLabel', etc.
    3. Vamos ler apenas a 'preferredLabel'
    """
    print(f"Lendo CSV real: {file_path}...")
    try:
        # Lê em chunks para não estourar a memória se o CSV for gigante
        # Supondo que o CSV tenha uma coluna chamada 'preferredLabel'
        chunk_size = 100
        batch = []
        
        # Simulação de leitura (troque pelo pd.read_csv real abaixo)
        # df = pd.read_csv(file_path)
        # for index, row in df.iterrows(): ...
        pass 
        
        print("Atenção: Função de CSV real pronta mas aguardando arquivo.")
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")

# --- Busca com Score ---

def buscar_skill_com_score(session, termo_sujo: str) -> Tuple[EscoSkill, float]:
    vetor_busca = get_embedding(termo_sujo)
    
    # O Pulo do Gato:
    # Solicitamos a distância explicitamente no SELECT
    distancia_col = EscoSkill.embedding.cosine_distance(vetor_busca).label("distancia")
    
    resultado = session.query(EscoSkill, distancia_col)\
        .order_by(distancia_col)\
        .limit(1)\
        .first()
    
    if resultado:
        skill_obj, distancia_valor = resultado
        # Converter Distância (0 a 2) em Similaridade (0 a 1)
        # Cosine Distance: 0 = igual, 1 = ortogonal, 2 = oposto
        # Simplificação útil: Similaridade ~ 1 - distancia
        similaridade = 1 - distancia_valor
        return skill_obj, similaridade
    
    return None, 0.0

# --- Execução ---

def main():
    setup_database()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Use seed_mock_data(session) OU ingest_real_esco_csv(session)
    seed_mock_data(session) 
    
    print("\n" + "="*60)
    print("TESTE DE ROBUSTEZ (Score & Threshold)")
    print("="*60)

    # Definindo um limiar de corte (Threshold)
    # Abaixo de 0.5 geralmente é ruído ou associação muito fraca
    THRESHOLD_ACEITAVEL = 0.5 

    entradas = [
        "sei mexer com a empilhadeira",   # Deve ter score alto
        "trabalhei soldando ferro",       # Deve ter score alto
        "adoro jogar futebol",            # Deve ter score BAIXO (Ruído)
        "sou muito proativo",             # Soft skill genérica (risco de alucinação)
        "python developer"                # Deve bater com dev python
    ]

    for entrada in entradas:
        match, score = buscar_skill_com_score(session, entrada)
        
        print(f"\nEntrada: '{entrada}'")
        
        if match:
            status = "✅ ACEITO" if score >= THRESHOLD_ACEITAVEL else "❌ REJEITADO (Ruído)"
            print(f"   -> Match Sugerido: {match.termo}")
            print(f"   -> Confiança (0-1): {score:.4f} | {status}")
        else:
            print("   -> Nenhum match encontrado")

if __name__ == "__main__":
    main()