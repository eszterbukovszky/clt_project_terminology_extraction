"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Hauptprogramm zur Terminologieextraktion und Evaluation
"""

from file_loader import CorpusLoader
from preprocessor import Preprocessor
from extraction import CandidateSelector
from scorer import TermScorer
from result_writer import ResultWriter

from collections import Counter
from typing import Set

def load_gold_terms(filepath: str) -> Set[str]:
    """Lädt die goldenen Fachbegriffe aus Datei."""
    with open(filepath, encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def compute_precision_recall(predicted: Set[str], gold: Set[str]):
    """Berechnet Precision und Recall."""
    true_positives = predicted & gold
    precision = len(true_positives) / len(predicted) if predicted else 0
    recall = len(true_positives) / len(gold) if gold else 0
    return precision, recall

def main():
    print("Lade Korpora...")
    acl = CorpusLoader.load_acl_abstracts("acl_abstracts")
    reuters = CorpusLoader.load_reuters_documents()

    print(f"ACL Abstracts: {len(acl)}, Reuters-Dokumente: {len(reuters)}")

    print("Extrahiere Kandidaten...")
    selector = CandidateSelector()
    domain_counts, domain_doc_counts = selector.extract_candidates(acl)

    print("Extrahiere Kandidaten aus Referenzkorpus...")
    pre = Preprocessor()
    all_bigrams_ref = []
    for doc in reuters:
        bigrams = pre.preprocess(doc)
        all_bigrams_ref.extend(bigrams)
    ref_counts = Counter(all_bigrams_ref)

    # Lade Goldstandard
    gold_terms = load_gold_terms("gold_terminology.txt")

    # Ergebnis-Ausgabe
    writer = ResultWriter()

    # 10 α/θ-Kombinationen
    combinations = [
        (0.0, 1.0),
        (0.2, 0.4),
        (0.3, 1.3),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.5, 1.1),
        (0.7, 0.7),
        (0.7, 1.2),
        (0.9, 0.8),
        (1.0, 0.9)
    ]

    for alpha, theta in combinations:
        print(f"\n--- Kombination: alpha={alpha}, theta={theta} ---")
        scorer = TermScorer(domain_counts, ref_counts, domain_doc_counts, alpha)
        scored_terms = scorer.score_all_terms(theta)

        writer.write_result_file(alpha, theta, scored_terms)

        predicted_terms = set(scored_terms.keys())
        precision, recall = compute_precision_recall(predicted_terms, gold_terms)
        print(f"Precision: {precision:.7f}, Recall: {recall:.7f}")

if __name__ == "__main__":
    main()
