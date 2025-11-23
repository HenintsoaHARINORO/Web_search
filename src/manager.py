import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from config import PORTFOLIO_FILE


class PortfolioManager:
    """Gestionnaire de portefeuille d'entreprises"""

    def __init__(self, filename: str = PORTFOLIO_FILE):
        self.filename = filename
        self.fieldnames = [
            'company_name',
            'resume',
            'comments'
        ]
        self._initialize_csv()

    def _initialize_csv(self):
        # création du fichier
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def company_exists(self, company_name: str) -> bool:
        companies = self.get_all_companies()
        return any(c['company_name'].lower() == company_name.lower() for c in companies)

    def add_company(self, company_name: str, resume: str, initial_comment: str = ""):
        if self.company_exists(company_name):
            return False

        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'company_name': company_name,
                'resume': resume,
                'comments': initial_comment
            })
        return True

    def add_comment(self, company_name: str, new_comment: str):
        companies = self.get_all_companies()
        found = False

        for company in companies:
            if company['company_name'].lower() == company_name.lower():
                found = True
                # Ajoute du nouveau commentaire
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                separator = " | " if company['comments'] else ""
                company['comments'] = f"{company['comments']}{separator}[{timestamp}] {new_comment}"
                break

        if not found:
            return False

        self._write_all_companies(companies)
        return True

    def get_last_comment(self, company_name: str) -> Optional[str]:
        """Retourne le dernier commentaire sans la date"""
        company = self.get_company(company_name)
        if not company or not company['comments']:
            return None

        # Sépare les commentaires par ' | '
        comments_list = company['comments'].split(' | ')

        if not comments_list:
            return None

        # Prend le dernier commentaire
        last_comment = comments_list[-1]

        # Enlève la date entre crochets [YYYY-MM-DD HH:MM]
        if ']' in last_comment:
            return last_comment.split('] ', 1)[-1]

        return last_comment

    def get_company(self, company_name: str) -> Optional[Dict]:
        companies = self.get_all_companies()
        for company in companies:
            if company['company_name'].lower() == company_name.lower():
                return company
        return None

    def get_all_companies(self) -> List[Dict]:
        companies = []
        if not os.path.exists(self.filename):
            return companies

        with open(self.filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            companies = list(reader)
        return companies

    def _write_all_companies(self, companies: List[Dict]):
        with open(self.filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(companies)