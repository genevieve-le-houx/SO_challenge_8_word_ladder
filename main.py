# https://www.datacamp.com/fr/tutorial/dijkstra-algorithm-in-python

from collections import Counter
from heapq import heapify, heappop, heappush
from typing import Tuple, List, Dict
from pathlib import Path

from tqdm import tqdm
import numpy as np


class WordsGraph:
    """
    Graph for the Dijkstra's algorithm
    Each weight is implicitly 1

    A word is a neighbor if the difference between them is only one letter.
    My solution take into account if we allow shuffling or not of letters.
    Building the graph is slow, so we can save the graph as a numpy matrix to reuse it
    """
    def __init__(self, words: List[str], shuffling: bool = False):
        self.words = words

        filename = "words_neighborhs_shuffle.npy" if shuffling else "words_neighborhs_no_shuffle.npy"
        filepath = Path(filename)

        if filepath.exists():
            self.words_neighbors = np.load(filepath)

        else:
            words_str = [np.array([x for x in word]) for word in words]

            self.words_neighbors = np.zeros((len(words), len(words)), dtype=int)

            print("Building graph")
            for i in tqdm(range(len(words))):
                for j in range(i+1, len(words)):
                    if i == j:
                        continue

                    self.words_neighbors[i][j] = self.words_is_neighbors(words_str[i], words_str[j], shuffling)

            self.words_neighbors = self.words_neighbors | self.words_neighbors.T

            np.save(filepath, self.words_neighbors)

    def get_words_no_neighborhs(self) -> List[str]:
        words_no_neighbors = []

        for word in self.words:
            word_neighbors = self.words_neighbors[self.words.index(word), :]

            if not np.any(word_neighbors):
                words_no_neighbors.append(word)

        return words_no_neighbors


    def shortest_distance(self, source: str) -> Tuple[Dict[str, float | int], Dict[str, str | None]]:
        distances = {word: float("inf") for word in self.words}
        distances[source] = 0

        pq = [(0, source)]
        heapify(pq)
        visited = set()

        while pq:
            current_distance, current_word = heappop(pq)

            if current_word in visited:
                continue

            visited.add(current_word)

            i_current_word = self.words.index(current_word)

            for i_neighbors in np.argwhere(self.words_neighbors[i_current_word, :]).flatten():
                neighbor_word = self.words[i_neighbors]

                tentative_distance = current_distance + 1
                if tentative_distance < distances[neighbor_word]:
                    distances[neighbor_word] = tentative_distance
                    heappush(pq, (tentative_distance, neighbor_word))

        predecessors: Dict[str, str | None] = {word: None for word in self.words}
        for word, distance in distances.items():
            i_current_word = self.words.index(word)

            for i_neighbors in np.argwhere(self.words_neighbors[i_current_word, :]).flatten():
                neighbor_word = self.words[i_neighbors]
                if distances[neighbor_word] == distance + 1:
                    predecessors[neighbor_word] = word

        return distances, predecessors


    def shortest_path(self, source: str, target: str) -> Tuple[int, List[str]]:
        """
        If there is no path, return -1, []
        :param source:
        :param target:
        :return:
        """
        _, predecessors = self.shortest_distance(source)

        path = []
        current_node = target

        while current_node:
            path.append(current_node)
            current_node = predecessors[current_node]
            if current_node is None:
                if path[-1] != source:
                    return -1, []

        path.reverse()

        return len(path), path


    @staticmethod
    def words_is_neighbors(a: np.ndarray, b: np.ndarray, shuffling: bool = False) -> bool:
        if shuffling:
            a_counter = dict(Counter(a).items())
            b_counter = dict(Counter(b).items())

            number_differences = 0
            for letter, count in a_counter.items():
                if letter in b_counter:
                    number_differences += abs(count - b_counter[letter])
                    b_counter.pop(letter)

                else:
                    number_differences += count

            # Letters left are the differences, but if there are more we add to the difference
            number_differences += abs(sum(b_counter.values()) - number_differences)

            return number_differences == 1
        else:
            return np.sum(a != b) == 1


def main():
    with open("sgb-words.txt", "r") as f:
        list_valid_words = [x.strip() for x in f.readlines()]

    input_pairs = [
        ("stone", "money"),
        ("bread", "crumb"),
        ("smile", "giant"),
        ("apple", "zebra"),
        ("other", "night"),
        ("bread", "blood"),
        ("black", "white")
    ]

    sufflings = [False, True]

    for shuffling in sufflings:
        text = "Shuffling" if shuffling else "No shuffling"
        print(f"--- {text} ---")

        words_graph = WordsGraph(list_valid_words, shuffling)

        words_no_neighbors = words_graph.get_words_no_neighborhs()
        print(f"Words without neighbors are: {words_no_neighbors}")

        for pair in input_pairs:
            ladder_length, ladder = words_graph.shortest_path(*pair)

            if ladder_length != -1:
                print(f"The ladder has a length of {ladder_length}: ", " -> ".join(ladder))
            else:
                print(f"There is no solution to the pair {pair}")

        print(" ")


if __name__ == '__main__':
    main()