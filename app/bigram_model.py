from collections import defaultdict, Counter
import random
import re


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """
        Initialize the BigramModel with a list of sentences (corpus).
        Builds vocabulary and bigram probabilities.
        """
        # Join all sentences into one text
        text = " ".join(corpus)

        # Tokenize text
        self.words = self.simple_tokenizer(text, frequency_threshold)

        # Build bigrams
        self.bigram_probs = self.build_bigram_probabilities(self.words)

    def simple_tokenizer(self, text, frequency_threshold=None):
        """
        Simple tokenizer that splits text into words (lowercase).
        Can filter out rare words if frequency_threshold is set.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())

        if not frequency_threshold:
            return tokens

        word_counts = Counter(tokens)
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]
        return filtered_tokens

    def build_bigram_probabilities(self, words):
        """
        Build bigram probabilities from a list of words.
        """
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            bigram_probs[w1][w2] = count / unigram_counts[w1]

        return bigram_probs

    def generate_text(self, start_word, num_words=20):
        """
        Generate text using the learned bigram probabilities.
        """
        current_word = start_word.lower()

        # If start word not in vocab, pick a random one
        if current_word not in self.bigram_probs:
            current_word = random.choice(list(self.bigram_probs.keys()))

        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break  # no continuation, stop early

            next_word = random.choices(
                list(next_words.keys()),
                weights=next_words.values()
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)
