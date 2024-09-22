# utils/data_processing.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import ngrams

def lexicon_dict(csvpath):
    """Load a lexicon from a CSV file and preprocess the text to create a dictionary of categories."""
    df = pd.read_csv(csvpath + "lex.csv")
    lexdict = {}
    for cat in df.columns:
        terms = df[cat].dropna().apply(lambda x: x.strip().lower().replace(" ", "_")).unique().tolist()
        expanded_terms = []
        for term in terms:
            parts = term.split("_")
            if len(parts) > 1:
                expanded_terms.extend(parts)
                expanded_terms.extend(["_".join(ngram) for ngram in ngrams(parts, 2)])
            else:
                expanded_terms.append(term)
        lexdict[cat] = list(set(expanded_terms))
    return lexdict

def allcountrylist():
    """Scrape a list of countries and their alternative names from Wikipedia."""
    URL = 'https://en.wikipedia.org/wiki/List_of_alternative_country_names'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    country_dict = {}
    tables = soup.findAll('table', attrs={'class': 'wikitable'})
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cols = [ele.text.strip() for ele in row.find_all('td')]
            if cols:
                primary_name = cols[1].split(" (")[0].strip()
                aliases = set([alias.split(" (")[0] for alias in cols[2].split(",") if "(" not in alias])
                country_dict[primary_name] = ", ".join(aliases)
    return country_dict

def famouscountry(csvpath, country_dict):
    """Load a list of famous countries and their alternative names, excluding some major countries."""
    df = pd.read_csv(csvpath + "famouscountry.csv")
    names = df['CountryName'].dropna().apply(str.strip).tolist()
    names = [name for name in names if name not in ['United States', 'China']]
    all_names_list = []
    for name in names:
        all_names_list.append(name.lower())
        all_names_list.extend([alias.lower() for alias in country_dict.get(name, "").split(", ")])
    return list(set(all_names_list))
