import os
from typing import List, Tuple
import requests
import re
from playwright.sync_api import sync_playwright
from config import GOOGLE_API_URL, GOOGLE_API_KEY, GOOGLE_CX, OLLAMA_API_URL, SCRAPE_MAX_CHARS, SCRAPE_TIMEOUT, REQUEST_TIMEOUT


def web_search(query: str, num_results: int = 1) -> Tuple[List[str], str]:
    """Search using Google Custom Search API - returns both URLs and snippets"""
    try:
        response = requests.get(
            GOOGLE_API_URL,
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": query,
                "num": num_results
            },
            timeout=REQUEST_TIMEOUT
        )

        data = response.json()

        if "error" in data:
            error_msg = f"Erreur de recherche: {data['error'].get('message', 'Erreur inconnue')}"
            return [], error_msg

        results = []
        urls = []
        for item in data.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url = item.get("link", "")
            results.append(f"{title}: {snippet}")
            urls.append(url)

        summary = "\n\n".join(results) if results else "Aucun résultat trouvé."
        return urls, summary

    except Exception as e:
        return [], f"Erreur de recherche: {str(e)}"


def scrape_url(url: str, max_chars: int = SCRAPE_MAX_CHARS) -> str:
    """Scrape a URL using Playwright and return the text content"""
    print(f"Scraping: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT REQUEST_TIMEOUT.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT)
            page.wait_for_timeout(2000)

            content = ""
            selectors = ["main", "article", "#content", ".content", "#main", "body"]

            for selector in selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        content = element.inner_text()
                        if len(content) > 200:
                            break
                except:
                    continue

            if not content:
                content = page.inner_text("body")

            browser.close()

            if len(content) > max_chars:
                content = content[:max_chars] + "..."

            return content.strip()

    except Exception as e:
        return f"Erreur de scraping: {str(e)}"


def clean_resume(text: str) -> str:
    """Nettoie le résumé généré par le LLM"""
    patterns_to_remove = [
        r"^Voici un résumé professionnel d[e'].*?:\s*",
        r"^Voici le résumé.*?:\s*",
        r"^Résumé professionnel.*?:\s*",
        r"^\*\*.*?\*\*\s*",
    ]

    cleaned = text.strip()
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def generate_resume(company_name: str, search_results: str, scraped_content: str = "") -> str:
    """Génère un résumé avec Ollama en utilisant les résultats de recherche ET le contenu scrapé"""

    context = f"Résultats de recherche:\n{search_results}"
    if scraped_content:
        context += f"\n\nContenu détaillé de la page web:\n{scraped_content}"

    messages = [
        {
            "role": "system",
            "content": """Tu es un assistant spécialisé dans l'analyse d'entreprises. 
Génère un résumé professionnel et structuré en 3-4 phrases maximum.
IMPORTANT: 
- Ne commence PAS par "Voici un résumé" ou des phrases d'introduction similaires.
- N'utilise PAS de mise en forme markdown (pas de ** ou ##).
- Commence directement par le contenu du résumé.
- Réponds TOUJOURS en français."""
        },
        {
            "role": "user",
            "content": f"Entreprise: {company_name}\n\n{context}\n\nRésume cette entreprise de manière claire et professionnelle."
        }
    ]

    try:
        payload = {
            "model": "llama3.2:3b",
            "messages": messages,
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        chunk = response.json()

        if chunk.get("message", {}).get("content"):
            raw_resume = chunk["message"]["content"]
            return clean_resume(raw_resume)
        else:
            return "Résumé non disponible"

    except Exception as e:
        return f"Erreur lors de la génération du résumé: {str(e)}"


def research_company(company_name: str, scrape_first: bool = True) -> Tuple[str, str, str]:
    """
    Recherche complète d'une entreprise: search + scrape + résumé
    Returns: (search_results, scraped_content, resume)
    """
    print(f"Recherche: {company_name}")

    # Step 1: Web search
    urls, search_results = web_search(company_name)
    print(f"Résultats de recherche: {search_results[:200]}...")

    # Step 2: Scrape first URL if enabled and available
    scraped_content = ""
    if scrape_first and urls:
        scraped_content = scrape_url(urls[0])
        print(f"Contenu scrapé: {scraped_content[:300]}...")

    # Step 3: Generate resume with all available info
    resume = generate_resume(company_name, search_results, scraped_content)
    print(f"Résumé: {resume}")

    return search_results, scraped_content, resume
