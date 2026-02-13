import time
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text
from sqlalchemy.orm import  declarative_base, sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

# --- Configuração ---
# Modelo Multilíngue (excelente para PT-BR e termos industriais)
print("Carregando modelo de IA (isso pode demorar na primeira vez)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
EMBEDDING_DIM = 384 # Dimensão deste modelo específico

# Conexão com o Banco do POC
DATABASE_URL = "postgresql://usuario:senha@localhost:5433/rh_poc"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# --- Modelagem ---

class EscoSkill(Base):
    """Tabela de Referência (Canonical Skills)"""
    __tablename__ = 'esco_skills'
    
    id = Column(Integer, primary_key=True)
    termo = Column(String, unique=True)
    embedding = Column(Vector(EMBEDDING_DIM)) 

    def __repr__(self):
        return f"<Skill: {self.termo}>"

# --- Funções de Apoio ---

def setup_database():
    """Habilita a extensão vector e cria tabelas"""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.drop_all(engine) # Limpa tudo para o teste
    Base.metadata.create_all(engine)

def get_embedding(text: str) -> List[float]:
    """Transforma texto em números (vetor)"""
    return model.encode(text).tolist()

def seed_esco_data(session):
    """
    Simula a importação da tabela ESCO.
    Na vida real, você leria um CSV com milhares de linhas.
    """
    print("Populando banco com taxonomias da ESCO...")
    skills_mock = [
        "Operador de Empilhadeira",
        "Soldagem MIG/MAG",
        "Controle de Qualidade Industrial",
        "Usinagem CNC",
        "Manutenção Elétrica Industrial",
        "Gestão de Equipe",
        "Desenvolvimento Python" # Só pra testar distorção
    ]
    
    objetos = []
    for skill in skills_mock:
        vetor = get_embedding(skill)
        objetos.append(EscoSkill(termo=skill, embedding=vetor))
    
    session.add_all(objetos)
    session.commit()
    print("Taxonomia ESCO carregada com sucesso!")

# --- O Coração da Solução: A Busca Semântica ---

def buscar_skill_mais_proxima(session, termo_sujo: str, threshold=0.4):
    """
    Recebe algo como 'manjo de solda' e acha a skill oficial.
    Usa Distância de Cosseno (<=>).
    """
    vetor_busca = get_embedding(termo_sujo)
    
    # Query mágica do PGVector:
    # Order by embedding <=> vetor_busca (ordena pela distância)
    resultado = session.query(EscoSkill)\
        .order_by(EscoSkill.embedding.cosine_distance(vetor_busca))\
        .limit(1)\
        .first()
        
    # Podemos calcular a distância para saber se é um match ruim
    # Nota: Em SQL puro seria SELECT 1 - (embedding <=> vetor) as similarity
    return resultado

# --- Execução do POC ---

def main():
    setup_database()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 1. Preparar o terreno
    seed_esco_data(session)
    
    print("\n" + "="*50)
    print("SIMULAÇÃO DE INGESTÃO DE CURRÍCULO (EXTRAÇÃO)")
    print("="*50)

    # 2. Simular o retorno da LLM (Strings sujas extraídas do PDF)
    # Cenário: Candidato escreveu de forma informal
    curriculo_skills_extraidas = [
        "sei mexer com a empilhadeira",
        "trabalhei soldando ferro",
        "fui lider de chão de fabrica",
        "adoro jogar futebol", # Ruído (não deve dar match bom)
        "sei mexer com python" # Distorção (não é skill industrial, mas tem relação com desenvolvimento)
    ]

    candidato_skills_finais = []

    for skill_suja in curriculo_skills_extraidas:
        match = buscar_skill_mais_proxima(session, skill_suja)
        
        print(f"\nEntrada do CV: '{skill_suja}'")
        print(f"Match na ESCO: '{match.termo}'")
        
        # Aqui, num sistema real, você verificaria a pontuação de similaridade.
        # Se for muito baixa, você descarta (ex: jogar futebol não bate com usinagem)
        candidato_skills_finais.append(match.termo)

    print("\n" + "="*50)
    print("RESULTADO FINAL NO PERFIL DO CANDIDATO")
    print("="*50)
    print(f"Skills normalizadas: {candidato_skills_finais}")
    
    # 3. Simular a Busca do Recrutador (Fase 2 do seu doc)
    print("\n" + "="*50)
    print("SIMULAÇÃO DE BUSCA DO RECRUTADOR")
    print("="*50)
    
    query_recrutador = "Preciso de um soldador experiente"
    print(f"Recrutador digitou: '{query_recrutador}'")
    
    match_recrutador = buscar_skill_mais_proxima(session, query_recrutador)
    print(f"Sistema sugeriu buscar candidatos com a skill: '{match_recrutador.termo}'")

if __name__ == "__main__":
    main()