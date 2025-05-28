"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Berechnung von Domain Relevance, Domain Consensus und Entscheidungspunktzahl
"""

import math
from collections import Counter
from typing import Dict


class TermScorer:
    """
    Berechnet Domain Relevance, Domain Consensus und Gesamtscore für die Begriffe.
    """

    def __init__(self, domain_counts: Counter, ref_counts: Counter,
                 doc_counts: Dict[str, Dict[int, int]], alpha: float):
        self.domain_counts = domain_counts
        self.ref_counts = ref_counts
        self.doc_counts = doc_counts
        self.alpha = alpha
        self.total_domain = sum(domain_counts.values())
        self.total_ref = sum(ref_counts.values())
        self.num_docs = self._get_num_docs()

    def _get_num_docs(self) -> int:
        """Anzahl der Dokumente aus der doc_counts-Struktur."""
        return max(max(doc_freqs.keys(), default=0) for doc_freqs in self.doc_counts.values()) + 1

    def _compute_domain_relevance(self, term: str) -> float:
        """Berechnet DRt, Kdom."""
        p_domain = self.domain_counts[term] / self.total_domain if self.total_domain > 0 else 0
        p_ref = self.ref_counts.get(term, 0) / self.total_ref if self.total_ref > 0 else 0

        denominator = p_domain + p_ref
        if denominator == 0:
            return 0.0
        return p_domain / denominator

    def _compute_domain_consensus(self, term: str) -> float:
        """Berechnet DCt, Kdom als Entropie."""
        doc_freqs = self.doc_counts[term]
        total = sum(doc_freqs.values())

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in doc_freqs.values():
            p = count / total
            if p > 0:
                entropy += p * math.log(1 / p, 2)  # log zur Basis 2
        return entropy

    def score_term(self, term: str) -> float:
        """Berechnet f(t) = α·DR + (1-α)·DC."""
        dr = self._compute_domain_relevance(term)
        dc = self._compute_domain_consensus(term)
        return self.alpha * dr + (1 - self.alpha) * dc

    def score_all_terms(self, threshold: float) -> Dict[str, float]:
        """
        Berechnet alle Scores und filtert nach θ.
        :param threshold: θ-Wert
        :return: Fachbegriffe mit Punktzahl
        """
        results = {}
        for term in self.domain_counts:
            score = self.score_term(term)
            if score > threshold:
                results[term] = score
        return results


if __name__ == "__main__":

    test_domain = Counter({"anaphora resolution": 10, "language model": 5})
    test_ref = Counter({"language model": 2})
    test_docs = {
        "anaphora resolution": {0: 6, 1: 4},
        "language model": {0: 3, 1: 2}
    }

    scorer = TermScorer(test_domain, test_ref, test_docs, alpha=0.5)
    result = scorer.score_all_terms(threshold=0.3)
    for term, score in result.items():
        print(f"{term}: {score:.4f}")
