"""
Autorin: Eszter Bukovszky
Datum: 31.03.2025
Zweck: Textvorverarbeitung inklusive Bigrams, Tokenisierung und Stoppwortentfernung
"""

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, bigrams
from typing import List
import string
import re

nltk.download('punkt')
nltk.download('stopwords')


class Preprocessor:
    """
    Führt die Vorverarbeitung eines Textes durch:
    - Tokenisierung
    - Entfernung von Satzzeichen
    - Erstellung von Bigrams
    - Entfernung von Stoppwörtern (nach der Bigram-Erstellung)
    """

    def __init__(self, language: str = 'english'):
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def tokenize(self, text: str) -> List[str]:
        """Tokenisiert den Text."""
        return word_tokenize(text)

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Entfernt reine Satzzeichen."""
        return [token for token in tokens if token not in self.punctuation]

    def remove_non_alphanumeric(self, tokens: List[str]) -> List[str]:
        """Entfernt Tokens, die keine Wörter oder Zahlen enthalten."""
        return [token for token in tokens if re.match(r"^\w+$", token)]

    def generate_bigrams(self, tokens: List[str]) -> List[str]:
        """Erstellt Bigrams als Liste von kleingeschriebenen Strings."""
        return [' '.join(bigram).lower() for bigram in bigrams(tokens)]

    def remove_stopwords_from_bigrams(self, bigram_list: List[str]) -> List[str]:
        """Entfernt Bigrams, in denen mindestens ein Wort ein Stoppwort ist."""
        return [bigram for bigram in bigram_list
                if all(word.lower() not in self.stop_words for word in bigram.split())]

    def preprocess(self, text: str) -> List[str]:
        """
        Gesamte Pipeline: Tokenisierung, Satzzeichen entfernen, Nicht-Wörter entfernen,
        Bigrams, Stoppwörter filtern
        """
        tokens = self.tokenize(text)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_non_alphanumeric(tokens)
        bigrams_raw = self.generate_bigrams(tokens)
        filtered_bigrams = self.remove_stopwords_from_bigrams(bigrams_raw)
        return filtered_bigrams


if __name__ == "__main__":
    # Beispiel
    sample_text = "Anaphora resolution is a task (in NLP), not a punctuation: task."
    pre = Preprocessor()
    processed = pre.preprocess(sample_text)
    print("Gefilterte Bigrams:", processed)

    # Test
    assert "anaphora resolution" in processed, "Test fehlgeschlagen"
    assert ") ." not in processed, "Satzzeichen-Bigramm nicht entfernt!"
    print("Unit-Test bestanden.")
