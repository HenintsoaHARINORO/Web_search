import csv
import os
from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from config import VECTOR_STORE_PATH, OLLAMA_MODEL, OLLAMA_API, PORTFOLIO_FILE


class PortfolioRAG:
    # RAG pour chercher et naviguer dans les portfolios
    def __init__(self, csv_file: str = PORTFOLIO_FILE, vector_store_path: str = VECTOR_STORE_PATH):
        self.csv_file = csv_file
        self.vector_store_path = vector_store_path
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API
        )
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API,
            temperature=0.3
        )
        self.vectorstore = None
        self.qa_chain = None

    def load_portfolio_data(self) -> List[Document]:
        documents = []

        if not os.path.exists(self.csv_file):
            print(f"❌ Fichier {self.csv_file} non trouvé")
            return documents

        with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Créer un contenu structuré pour chaque entreprise
                content = f"""
Entreprise: {row['company_name']}
Date d'ajout: {row['search_date']}
Dernière mise à jour: {row['last_updated']}

Résumé:
{row['resume']}

Commentaires:
{row['comments'] if row['comments'] else 'Aucun commentaire'}
"""

                doc = Document(
                    page_content=content,
                    metadata={
                        "company_name": row['company_name'],
                        "search_date": row['search_date'],
                        "last_updated": row['last_updated'],
                        "source": "portfolio_csv"
                    }
                )
                documents.append(doc)

        print(f"{len(documents)} entreprise(s) chargée(s)")
        return documents

    def build_vectorstore(self, force_rebuild: bool = False):
        """Construit ou charge le vector store FAISS"""

        # Vérifier si un vector store existe déjà
        if os.path.exists(self.vector_store_path) and not force_rebuild:
            print("Chargement du vector store existant...")
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Vector store chargé")
        else:
            print("Construction du vector store...")
            documents = self.load_portfolio_data()

            if not documents:
                print("Aucune donnée à indexer")
                return

            # Créer le vector store
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )

            # Sauvegarder pour une utilisation future
            self.vectorstore.save_local(self.vector_store_path)
            print(f"Vector store créé et sauvegardé dans {self.vector_store_path}")

    def rebuild_index(self):
        """Force la reconstruction de l'index (à appeler après mise à jour du CSV)"""
        print("Reconstruction de l'index...")
        self.build_vectorstore(force_rebuild=True)

    def setup_qa_chain(self):
        """Configure la chaîne de questions-réponses"""
        if self.vectorstore is None:
            print("Vector store non initialisé")
            return

        # Template de prompt personnalisé
        template = """Tu es un assistant spécialisé dans l'analyse de portefeuille d'entreprises.
Utilise les informations suivantes pour répondre à la question de manière précise et professionnelle.

Contexte:
{context}

Question: {question}

Instructions:
- Si la question concerne une entreprise spécifique, donne tous les détails pertinents
- Si la question est générale, résume les entreprises pertinentes
- Si l'information n'est pas dans le contexte, dis-le clairement
- Réponds toujours en français

Réponse:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # nombre de documents à retourner
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        print("Chaîne QA configurée")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Recherche par similarité dans le vector store"""
        if self.vectorstore is None:
            print("Vector store non initialisé")
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def ask(self, question: str) -> Dict:
        """Pose une question sur le portefeuille"""
        if self.qa_chain is None:
            print("Chaîne QA non initialisée")
            return {"answer": "Système non initialisé", "sources": []}

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

    def list_all_companies(self) -> List[str]:
        """Liste toutes les entreprises du portefeuille"""
        documents = self.load_portfolio_data()
        return [doc.metadata["company_name"] for doc in documents]

    def find_companies_by_keyword(self, keyword: str) -> List[Document]:
        """Trouve les entreprises contenant un mot-clé"""
        return self.search(keyword, k=5)


def initialize_rag(rebuild: bool = False) -> PortfolioRAG:
    """Initialise le système RAG"""
    rag = PortfolioRAG()
    rag.build_vectorstore(force_rebuild=rebuild)
    rag.setup_qa_chain()
    return rag


# Exemples d'utilisation
if __name__ == "__main__":
    print("=" * 80)
    print("SYSTÈME RAG - PORTFOLIO D'ENTREPRISES")
    print("=" * 80)

    # Initialiser le RAG
    rag = initialize_rag(rebuild=True)  # rebuild=True pour forcer la reconstruction

    # Exemple 1: Liste des entreprises
    print("\nListe des entreprises:")
    companies = rag.list_all_companies()
    for company in companies:
        print(f"   - {company}")

    # Exemple 2: Recherche par similarité
    print("\nRecherche par mot-clé 'formation':")
    results = rag.find_companies_by_keyword("formation")
    for i, doc in enumerate(results, 1):
        print(f"\n   Résultat {i}: {doc.metadata['company_name']}")
        print(f"   Extrait: {doc.page_content[:200]}...")

    # Exemple 3: Questions en langage naturel
    print("\nQuestions en langage naturel:")

    questions = [
        "Quelles entreprises dans mon portefeuille sont déjà contactées ?",
    ]

    for question in questions:
        print(f"\n Question: {question}")
        response = rag.ask(question)
        print(f"Réponse: {response['answer']}")
        print(f"Sources: {len(response['sources'])} document(s)")
