"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Laden des Domänen- und Referenzkorpus
"""

import os
from typing import List
import bibtexparser
from nltk.corpus import reuters
import nltk

nltk.download('reuters')


class CorpusLoader:
    """
    Lädt ACL-Abstracts aus .bib und das Reuters-Korpus aus NLTK.
    """

    @staticmethod
    def load_acl_abstracts(bib_path: str) -> List[str]:
        """
        Lädt ACL-Abstracts aus einer .bib-Datei.
        :param bib_path: Pfad zur .bib-Datei
        :return: Liste der Abstracts
        """
        with open(bib_path, encoding="utf-8") as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        abstracts = []
        for entry in bib_database.entries:
            if 'abstract' in entry:
                abstracts.append(entry['abstract'])
        return abstracts

    @staticmethod
    def load_reuters_documents() -> List[str]:
        """
        Lädt die Reuters-Dokumente als Textstrings.
        :return: Liste von Reuters-Texten
        """
        doc_ids = reuters.fileids()
        texts = [' '.join(reuters.words(doc_id)) for doc_id in doc_ids]
        return texts


if __name__ == "__main__":
    # Tests
    reuters_docs = CorpusLoader.load_reuters_documents()
    print("Beispiel Reuters-Dokument:", reuters_docs[0][:300])

    acl_abstracts = CorpusLoader.load_acl_abstracts("acl_abstracts")
    print("ACL Abstracts geladen:", len(acl_abstracts))
