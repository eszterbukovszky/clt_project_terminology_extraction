"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Auswahl und Zählung von Kandidatenbegriffen aus dem Domänenkorpus
"""

from collections import Counter, defaultdict
from typing import List, Dict
from preprocessor import Preprocessor


class CandidateSelector:
    """
    Extrahiert Bigram-Kandidatenbegriffe und zählt deren Häufigkeiten.
    """

    def __init__(self):
        self.preprocessor = Preprocessor()

    def extract_candidates(self, corpus: List[str]) -> (Counter, Dict[str, Dict[int, int]]):
        """
        Extrahiert gültige Bigrams und zählt globale Häufigkeit über das ganze Korpus und Häufigkeit pro Dokument.
        :param corpus: Liste von Dokumenten aus dem Domänenkorpus
        :return: Tuple:
            - Counter: Begriff -< Gesamthäufigkeit (c(t, Kdom))
            - Dict: Begriff -< {Dokumentindex: Häufigkeit in diesem Dokument} (für DC)
        """
        total_counts = Counter()
        per_doc_counts = defaultdict(lambda: defaultdict(int))

        for doc_idx, doc in enumerate(corpus):
            bigrams = self.preprocessor.preprocess(doc)
            doc_counter = Counter(bigrams)

            for bigram, count in doc_counter.items():
                total_counts[bigram] += count
                per_doc_counts[bigram][doc_idx] = count

        return total_counts, per_doc_counts


if __name__ == "__main__":
    # Test mit ACL Abstracts
    from file_loader import CorpusLoader

    abstracts = CorpusLoader.load_acl_abstracts("acl_abstracts")
    selector = CandidateSelector()
    total_counts, per_doc_counts = selector.extract_candidates(abstracts)

    print("Top 5 häufigste Begriffe:")
    for term, count in total_counts.most_common(5):
        print(f"{term}: {count}")

    print("\nBeispiel für DC-Zählung:")
    example_term = list(per_doc_counts.keys())[0]
    print(f"{example_term} -< Verteilung:", dict(per_doc_counts[example_term]))
