import csv, json
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
        self.metadata_file = os.path.join(vector_store_path, "index_metadata.json")
        self.embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_API)
        self.llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_API, temperature=0.3)
        self.vectorstore = None
        self.qa_chain = None

    def _get_csv_modification_time(self) -> float:
        """Get the last modification time of the CSV file."""
        if os.path.exists(self.csv_file):
            return os.path.getmtime(self.csv_file)
        return 0

    def _load_index_metadata(self) -> dict:
        """Load metadata about the last indexation."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"last_csv_mtime": 0, "indexed_companies": []}

    def _update_vectorstore_incrementally(self):
        """Identify new documents and merge them with the existing index."""
        metadata = self._load_index_metadata()
        existing_company_names = set(metadata.get("indexed_companies", []))

        # Also check from vectorstore if metadata is empty
        if not existing_company_names and self.vectorstore:
            for doc_id in self.vectorstore.docstore._dict:
                doc = self.vectorstore.docstore.lookup(doc_id)
                if doc and doc.metadata.get("company_name"):
                    existing_company_names.add(doc.metadata["company_name"])

        all_documents = self.load_portfolio_data()
        new_documents = [
            doc for doc in all_documents
            if doc.metadata["company_name"] not in existing_company_names
        ]

        if not new_documents:
            print("Aucune nouvelle donnée à indexer. Le vector store est à jour.")
            return

        print(f"Indexation de {len(new_documents)} nouveau(x) document(s)...")
        new_faiss_index = FAISS.from_documents(new_documents, self.embeddings)
        self.vectorstore.merge_from(new_faiss_index)
        self.vectorstore.save_local(self.vector_store_path)

        # Update metadata with all companies
        all_company_names = [doc.metadata["company_name"] for doc in all_documents]
        self._save_index_metadata(all_company_names)
        print(f"Fusion réussie : {len(new_documents)} documents ajoutés.")

    def load_portfolio_data(self) -> List[Document]:
        documents = []

        if not os.path.exists(self.csv_file):
            print(f"Fichier {self.csv_file} non trouvé")
            return documents

        with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Extraire le dernier commentaire sans date
                comments_text = row['comments'] if row['comments'] else 'Aucun commentaire'

                # Si des commentaires existent, extraire le dernier sans date
                if comments_text and comments_text != 'Aucun commentaire':
                    comments_list = comments_text.split(' | ')
                    last_comment = comments_list[-1]

                    # Enlever la date [YYYY-MM-DD HH:MM] si présente
                    if ']' in last_comment:
                        last_comment_clean = last_comment.split('] ', 1)[-1]
                    else:
                        last_comment_clean = last_comment

                    comments_display = f"Dernier commentaire: {last_comment_clean}"
                else:
                    comments_display = 'Aucun commentaire'

                # Créer un contenu structuré pour chaque entreprise
                content = f"""
    Entreprise: {row['company_name']}

    Résumé:
    {row['resume']}

    Commentaires:
    {comments_display}
    """

                doc = Document(
                    page_content=content,
                    metadata={
                        "company_name": row['company_name'],
                        "source": "portfolio_csv"
                    }
                )
                documents.append(doc)

        print(f"{len(documents)} entreprise(s) chargée(s)")
        return documents

    def _save_index_metadata(self, indexed_companies: List[str]):
        """Save metadata about the current indexation."""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        metadata = {
            "last_csv_mtime": self._get_csv_modification_time(),
            "indexed_companies": indexed_companies
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def _needs_update(self) -> bool:
        """Check if the CSV file was modified since last indexation."""
        metadata = self._load_index_metadata()
        current_mtime = self._get_csv_modification_time()
        return current_mtime > metadata["last_csv_mtime"]

    def build_vectorstore(self, force_rebuild: bool = False):
        """Build or load the FAISS vector store with automatic update detection."""
        if not os.path.exists(self.csv_file):
            print(f"Fichier source CSV {self.csv_file} non trouvé.")
            return

        vectorstore_exists = os.path.exists(self.vector_store_path)

        # Check if CSV was modified BEFORE loading vector store
        needs_update = self._needs_update() if vectorstore_exists else False

        # If CSV changed, force a complete rebuild
        if needs_update:
            print("Modifications détectées dans le CSV, reconstruction complète...")
            force_rebuild = True

        if vectorstore_exists and not force_rebuild:
            print("Chargement du vector store existant...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Vector store chargé et à jour.")
                return
            except Exception as e:
                print(f"Erreur lors du chargement : {e}. Reconstruction forcée...")
                force_rebuild = True

        # Full rebuild (either first time or after detecting changes)
        if not vectorstore_exists or force_rebuild:
            print("Construction complète du vector store...")
            documents = self.load_portfolio_data()
            if not documents:
                print("Aucune donnée à indexer")
                return

            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.vectorstore.save_local(self.vector_store_path)

            # Save metadata with current timestamp
            company_names = [doc.metadata["company_name"] for doc in documents]
            self._save_index_metadata(company_names)
            print(f"Vector store créé avec {len(documents)} entreprise(s) dans {self.vector_store_path}")

    def rebuild_index(self):
        """Force la reconstruction de l'index (à appeler après mise à jour du CSV)"""
        print("Reconstruction de l'index...")
        self.build_vectorstore(force_rebuild=True)

    def setup_qa_chain(self):
        """Configure la chaîne de questions-réponses"""
        if self.vectorstore is None:
            print("Vector store non initialisé")
            return
        num_docs = len(self.vectorstore.docstore._dict)
        k_value = max(num_docs, 2)
        # Template de prompt personnalisé
        template = """Tu es un assistant spécialisé dans l'analyse de portefeuille d'entreprises.
Utilise les informations suivantes pour répondre à la question de manière précise et professionnelle.

Contexte:
{context}

Question: {question}

Instructions:
- Si la question concerne une entreprise spécifique, donne tous les détails pertinents
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
                search_kwargs={"k": k_value}  # nombre de documents à retourner
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
