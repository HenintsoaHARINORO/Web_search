import streamlit as st
from src.manager import PortfolioManager
from src.web_search import web_search, generate_resume
from src.retrieval import initialize_rag

# Configuration de la page
st.set_page_config(
    page_title="Assistant Portfolio",
    page_icon="",
    layout="centered"
)

# Initialisation du state
if "started" not in st.session_state:
    st.session_state.started = False
if "step" not in st.session_state:
    st.session_state.step = "welcome"
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "current_resume" not in st.session_state:
    st.session_state.current_resume = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = PortfolioManager()
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_to_menu():
    """Retourne au menu principal"""
    st.session_state.step = "menu"
    st.session_state.current_company = None
    st.session_state.current_resume = None


def main():
    st.title("Assistant Portfolio d'Entreprises")

    # Écran de démarrage
    if not st.session_state.started:
        st.markdown("---")
        st.markdown("### Bienvenue!")
        st.markdown("Cet assistant vous aide à gérer votre portefeuille d'entreprises.")

        if st.button("Démarrer", type="primary", use_container_width=True):
            st.session_state.started = True
            st.session_state.step = "welcome"
            st.rerun()
        return

    # Message de bienvenue
    if st.session_state.step == "welcome":
        st.info("Je suis votre assistante de recherche de portfolio. Comment puis-je vous aider?")
        st.session_state.step = "menu"

    # Menu principal
    if st.session_state.step == "menu":
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Rechercher une entreprise", use_container_width=True):
                st.session_state.step = "search_company"
                st.rerun()

        with col2:
            if st.button("Discuter avec le portfolio", use_container_width=True):
                st.session_state.step = "chat"
                # Initialiser le RAG si pas encore fait
                if st.session_state.rag is None:
                    with st.spinner("Initialisation du système RAG..."):
                        st.session_state.rag = initialize_rag(rebuild=True)
                st.rerun()

        with col3:
            if st.button("📋 Voir le portfolio", use_container_width=True):
                st.session_state.step = "view_portfolio"
                st.rerun()

    # Recherche d'entreprise
    elif st.session_state.step == "search_company":
        st.markdown("---")
        st.subheader("Rechercher une entreprise")

        company_name = st.text_input("Nom de l'entreprise:", placeholder="")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Rechercher", type="primary", use_container_width=True):
                if company_name:
                    with st.spinner(f"Recherche en cours pour '{company_name}'..."):
                        search_results = web_search(company_name)
                        resume = generate_resume(company_name, search_results)

                        st.session_state.current_company = company_name
                        st.session_state.current_resume = resume
                        st.session_state.step = "show_resume"
                        st.rerun()
                else:
                    st.warning("Veuillez entrer un nom d'entreprise.")

        with col2:
            if st.button("Retour", use_container_width=True):
                reset_to_menu()
                st.rerun()

    # Affichage du résumé
    elif st.session_state.step == "show_resume":
        st.markdown("---")
        st.subheader(f"Résumé: {st.session_state.current_company}")

        st.success(st.session_state.current_resume)

        # Vérifier si l'entreprise existe déjà
        if st.session_state.portfolio.company_exists(st.session_state.current_company):
            st.info("Cette entreprise existe déjà dans votre portfolio.")
        else:
            if st.button("➕ Ajouter au portfolio", type="primary"):
                st.session_state.step = "add_comment"
                st.rerun()

        st.markdown("---")
        if st.button("Retour au menu", use_container_width=True):
            reset_to_menu()
            st.rerun()

    # Ajout de commentaire
    elif st.session_state.step == "add_comment":
        st.markdown("---")
        st.subheader(f"Ajouter un commentaire pour {st.session_state.current_company}")

        st.info(f"**Résumé:** {st.session_state.current_resume}")

        comment = st.text_area("Commentaire (optionnel):", placeholder="")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Sauvegarder", type="primary", use_container_width=True):
                success = st.session_state.portfolio.add_company(
                    st.session_state.current_company,
                    st.session_state.current_resume,
                    comment
                )
                if success:
                    st.success(f"{st.session_state.current_company} ajouté au portfolio!")
                    # Reconstruire l'index RAG
                    if st.session_state.rag:
                        st.session_state.rag.rebuild_index()
                    st.session_state.step = "saved"
                    st.rerun()
                else:
                    st.error("Erreur lors de l'ajout.")

        with col2:
            if st.button("Retour", use_container_width=True):
                st.session_state.step = "show_resume"
                st.rerun()

    # Confirmation de sauvegarde
    elif st.session_state.step == "saved":
        st.markdown("---")
        st.success(f"L'entreprise **{st.session_state.current_company}** a été ajoutée avec succès!")

        if st.button("Retour au menu", type="primary", use_container_width=True):
            reset_to_menu()
            st.rerun()

    # Vue du portfolio
    elif st.session_state.step == "view_portfolio":
        st.markdown("---")
        st.subheader("Mon Portfolio")

        companies = st.session_state.portfolio.get_all_companies()

        if not companies:
            st.warning("Votre portfolio est vide. Commencez par ajouter des entreprises!")
        else:
            for i, company in enumerate(companies):
                with st.expander(f"{company['company_name']}", expanded=False):
                    st.markdown(f"Résumé: {company['resume']}")
                    st.markdown(f"Commentaires: {company['comments'] if company['comments'] else 'Aucun'}")

                    # Option pour ajouter un commentaire
                    new_comment = st.text_input(f"Ajouter un commentaire:", key=f"comment_{i}")
                    if st.button("Ajouter", key=f"btn_{i}"):
                        if new_comment:
                            st.session_state.portfolio.add_comment(company['company_name'], new_comment)
                            st.success("Commentaire ajouté!")
                            st.rerun()

        st.markdown("---")
        if st.button("Retour au menu", use_container_width=True):
            reset_to_menu()
            st.rerun()

    # Chat avec le portfolio (RAG)
    elif st.session_state.step == "chat":
        st.markdown("---")
        st.subheader("Discuter avec le Portfolio")

        # Afficher l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input utilisateur
        if prompt := st.chat_input("Posez une question sur votre portfolio..."):
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    if st.session_state.rag:
                        response = st.session_state.rag.ask(prompt)
                        answer = response["answer"]
                    else:
                        answer = "Le système RAG n'est pas initialisé."

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Effacer la conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Retour au menu", use_container_width=True):
                reset_to_menu()
                st.rerun()


if __name__ == "__main__":
    main()