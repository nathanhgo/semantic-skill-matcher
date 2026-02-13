import pandas as pd
from typing import List, Tuple
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

# --- Configuração ---
print("Carregando modelo de IA...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
EMBEDDING_DIM = 384 

DATABASE_URL = "postgresql://usuario:senha@localhost:5433/rh_poc"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# --- Modelagem ---
class EscoSkill(Base):
    __tablename__ = 'esco_skills'
    id = Column(Integer, primary_key=True)
    # Aumentei o tamanho da string pois algumas skills são longas
    termo = Column(String(500), unique=True) 
    embedding = Column(Vector(EMBEDDING_DIM)) 

    def __repr__(self):
        return f"<Skill: {self.termo}>"

# --- Funções de Banco ---
def setup_database():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

# --- Ingestão Otimizada (Batch Processing) ---
def ingest_real_esco_csv(session, file_path='skills_pt.csv'):
    print(f"Lendo CSV real: {file_path}...")
    
    try:
        # Carrega o CSV com Pandas
        df = pd.read_csv(file_path)
        
        # Filtra apenas a coluna principal (preferredLabel)
        # Atenção: Confirme se a coluna no seu CSV chama 'preferredLabel'
        if 'preferredLabel' not in df.columns:
            raise ValueError(f"Coluna 'preferredLabel' não encontrada. Colunas disponíveis: {df.columns}")
        
        # Remove duplicatas e valores nulos para não quebrar o banco
        termos_unicos = df['preferredLabel'].dropna().unique().tolist()
        
        total = len(termos_unicos)
        print(f"Encontradas {total} skills únicas. Iniciando vetorização em lote...")
        
        # CONFIGURAÇÃO DE LOTE (Batch Size)
        # Processar 100 por vez é muito mais rápido que 1 por 1
        batch_size = 64 
        
        for i in range(0, total, batch_size):
            lote_termos = termos_unicos[i : i + batch_size]
            
            # A mágica: A IA converte a lista inteira de uma vez
            lote_vetores = model.encode(lote_termos)
            
            # Prepara objetos para salvar no banco
            objetos_banco = []
            for termo, vetor in zip(lote_termos, lote_vetores):
                objetos_banco.append(EscoSkill(termo=termo, embedding=vetor.tolist()))
            
            # Salva no banco
            session.add_all(objetos_banco)
            session.commit()
            
            # Feedback visual (barra de progresso simples)
            print(f"Processado: {i + len(lote_termos)}/{total}", end='\r')
            
        print("\nSucesso! Todas as skills foram carregadas.")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{file_path}' não encontrado na pasta.")
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")

# --- Busca ---
def buscar_skill(session, termo_sujo: str):
    vetor = model.encode(termo_sujo).tolist()
    distancia = EscoSkill.embedding.cosine_distance(vetor).label("distancia")
    
    res = session.query(EscoSkill, distancia).order_by(distancia).limit(1).first()
    
    if res:
        skill, dist = res
        return skill.termo, 1 - dist
    return None, 0.0

# --- Main ---
def main():
    setup_database()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Mude para True para usar o CSV real
    USAR_CSV_REAL = True 
    
    if USAR_CSV_REAL:
        # Certifique-se que o arquivo 'skills_pt.csv' está na pasta
        ingest_real_esco_csv(session, 'skills_pt.csv')
    else:
        # Mock para teste rápido se não tiver o CSV
        print("Usando dados Mock...") 
        # ... (código do mock anterior) ...

    # Teste Final
    testes = [
        "manjo de solda elétrica",
        "sei programar em python",
        "operador de empilhadeira",
        "fiz curso de primeiros socorros"
    ]
    
    print("\n--- TESTE FINAL ---")
    for t in testes:
        termo, score = buscar_skill(session, t)
        print(f"Input: '{t}' \nMatch: '{termo}' (Confiança: {score:.2f})\n")

if __name__ == "__main__":
    main()