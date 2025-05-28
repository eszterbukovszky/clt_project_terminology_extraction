"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Ausgabe der finalen Ergebnisse in formatierte Dateien
"""

import os
from typing import Dict


class ResultWriter:
    """
    Speichert die ausgewählten Fachbegriffe mit Punktzahl in einer Datei.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_result_file(self, alpha: float, theta: float, scored_terms: Dict[str, float]):
        """
        Schreibt eine Datei mit zwei Kopfzeilen (alpha, theta) und
        tab-separierter Liste von Begriffen und Punktzahl.
        """
        filename = f"{self.output_dir}/terms_alpha{alpha:.2f}_theta{theta:.2f}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{alpha}\n")
            f.write(f"{theta}\n")
            for term, score in sorted(scored_terms.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{term}\t{score:.4f}\n")

        print(f"Ergebnis gespeichert: {filename}")


if __name__ == "__main__":
    print("Demonstration von ResultWriter.")

    test_results = {"anaphora resolution": 0.75, "language model": 0.42}

    alpha = 0.5
    theta = 0.3

    writer = ResultWriter("../../Documents/Egyetem/CLT/CLT-Projekt/output")
    writer.write_result_file(alpha, theta, test_results)

    expected_filename = f"output/terms_alpha{alpha:.2f}_theta{theta:.2f}.txt"

    print("Begriffe und Scores wurden gespeichert in:", expected_filename)

    # Datei öffnen und Inhalt prüfen
    with open(expected_filename, encoding="utf-8") as f:
        lines = f.readlines()
        assert lines[0].strip() == "0.5", "Alpha-Zeile falsch"
        assert lines[1].strip() == "0.3", "Theta-Zeile falsch"
        assert lines[2].strip() == "anaphora resolution\t0.7500", "Erster Begriff falsch"
        assert lines[3].strip() == "language model\t0.4200", "Zweiter Begriff falsch"
        print("Unit-Test bestanden: Datei-Inhalt stimmt.")


