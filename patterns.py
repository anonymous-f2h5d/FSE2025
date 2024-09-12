import json

import dataHelper
from dtwWrapper import DtwWrapper
from collections import defaultdict
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

class Patterns:

    def __init__(self):
        """
            Initializes the class by loading time series data, pattern labels, and preparing dictionaries
            for patterns and their details.
        """

        with open("data/softdtw/cluster_labels.json") as json_file:
            time_series = json.load(json_file)

        self.time_series_details = time_series['timeseries_details']
        self.time_series_labels = time_series['cluster_label']
        self.patterns_dict = self.prepare_pattern_series()[0]
        self.patterns_details_dict = self.prepare_pattern_series()[1]

    def prepare_pattern_series(self):
        """
            Prepares and organizes time series data based on their pattern labels.

            This method processes the time series data and their corresponding labels to create dictionaries
            that group time series patterns and their details by repository. It ensures that the number of labels
            matches the number of time series details before proceeding.
        """
        if len(self.time_series_labels) != len(self.time_series_details):
            raise AssertionError

        patterns_dict = {}
        patterns_details_dict = {}
        for i in range(0, len(self.time_series_details)):
            repo = self.time_series_details[i][0]
            pattern = self.time_series_labels[i]

            if repo not in patterns_dict:
                patterns_dict[repo] = []
                patterns_details_dict[repo] = []

            patterns_dict[repo].append(pattern)
            patterns_details_dict[repo].append(self.time_series_details[i])

        return patterns_dict, patterns_details_dict

    def print_transition_matrix(self):
        """
            Prints the transition matrix of patterns to the console.
            This method utilizes the `build_markov_chain` method to generate the Markov Chain transition matrix.
        """

        for current_pattern, transitions in self.build_markov_chain().items():
            print(f"From {current_pattern}:")
            for next_pattern, probability in transitions.items():
                print(f"  to {next_pattern} with probability {probability:.2f}")

    def build_markov_chain(self):
        """
            Constructs a Markov Chain transition matrix based on observed patterns in time series data.

            This method creates a transition matrix that captures the probability of transitioning from one pattern to
            another within the sequences of patterns provided. It counts the occurrences of transitions, converts
            these counts into probabilities, and then saves the result to a CSV file.
        """
        transition_matrix = defaultdict(lambda: defaultdict(int))

        for sequence in list(self.patterns_dict.values()):
            for i in range(len(sequence) - 1):
                current_pattern = sequence[i]
                next_pattern = sequence[i + 1]
                transition_matrix[dataHelper.rename_cluster(current_pattern)][dataHelper.rename_cluster(next_pattern)] += 1

        # Convert counts to probabilities
        for current_pattern, transitions in transition_matrix.items():
            total_transitions = sum(transitions.values())
            for next_pattern in transitions:
                transition_matrix[current_pattern][next_pattern] /= total_transitions

        df = pd.DataFrame.from_dict(transition_matrix, orient='index').fillna(0)
        df.index.name = 'From Pattern'
        df.columns.name = 'To Pattern'

        # Print the DataFrame
        print(df.round(2))
        df.to_csv('data/markov.csv')

        return transition_matrix

    def distributions_in_different_stages(self):
        """
            Analyzes and visualizes the distribution of refactoring release patterns across different stages based on
            their length.

            This method performs the following steps:

            1. **Calculate Quartiles:**
               - Determines the quartiles of pattern lengths to define chunks or stages.

            2. **Divide Patterns:**
               - Divides each pattern into segments corresponding to the calculated quartiles.

            3. **Aggregate Distributions:**
               - Aggregates the patterns in each stage and counts the frequency of each pattern in these stages.

            4. **Normalize Distributions:**
               - Normalizes the counts in each stage to ensure comparability between stages.

            6. **Plot the Data:**
               - Generates a line plot showing the distribution of refactoring release patterns across different stages. The plot is saved as both a PDF and PNG file.
        """

        patterns_2d = list(self.patterns_dict.values())

        # Step 1: Calculate quartiles based on array lengths
        array_lengths = [len(arr) for arr in patterns_2d]

        quartiles = np.percentile(array_lengths, [25, 50, 75])

        print(quartiles)
        chunks = [
            [0, int(quartiles[0])],
            [int(quartiles[0]), int(quartiles[1])],
            [int(quartiles[1]), int(quartiles[2])],
            [int(quartiles[2]), max(array_lengths)]
        ]

        distributions = [
            [],
            [],
            [],
            []
        ]

        for pattern in patterns_2d:
            chunked = [
                pattern[chunks[0][0]:chunks[0][1]],
                pattern[chunks[1][0]:chunks[1][1]],
                pattern[chunks[2][0]:chunks[2][1]],
                pattern[chunks[3][0]:chunks[3][1]]
                ]

            for i in range(0,4):
                distributions[i] = distributions[i]+chunked[i]


        distributions_count = {}
        from collections import Counter
        counts = []
        for i in range(0, 4):
            dist_count = dict(Counter(distributions[i]))
            dist_count = dict(sorted(dist_count.items()))

            distributions_count[str(i)] = dist_count
            counts.append(sum(dist_count.values()))

        normalize_value = int(min(counts))
        normalized_dict = {}

        for key, inner_dict in distributions_count.items():
            total = sum(inner_dict.values())
            normalized_inner_dict = {k: v / total * normalize_value for k, v in inner_dict.items()}
            normalized_dict[key] = normalized_inner_dict

        print(normalized_dict)
        normalized_array = []

        for column in normalized_dict.values():
            normalized_array.append(list(column.values()))

        # transposed_list = list(map(list, zip(*normalized_array)))
        transposed_list = normalized_array

        normalized_data = []

        for sublist in transposed_list:
            # Step 1: Calculate the sum of the sublist
            sublist_sum = sum(sublist)

            # Step 2 and 3: Normalize each element in the sublist
            normalized_sublist = [(x / sublist_sum) * 100 for x in sublist]

            normalized_data.append(normalized_sublist)

        import dataHelper
        to_plot = {
            dataHelper.rename_cluster(2): [sublist[2] for sublist in normalized_data],
            dataHelper.rename_cluster(1): [sublist[1] for sublist in normalized_data],
            dataHelper.rename_cluster(3): [sublist[3] for sublist in normalized_data],
            dataHelper.rename_cluster(0): [sublist[0] for sublist in normalized_data]
        }
        print(to_plot)
        plt.figure(figsize=(8, 6))
        x_points = ['Early', 'Middle', 'Late', 'Last']

        # Plotting the connected dot plot with raw values
        colors = ['blue', 'green', 'red', 'orange']

        # Plotting
        for (key, values), color in zip(to_plot.items(), colors):
            plt.plot(x_points, values, marker='o', label=key, linestyle='-', color=color)

        plt.legend()
        plt.xlabel('Stage')
        plt.ylabel('Values (Percenrage)')
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        plt.savefig('images/distributions.pdf', format='pdf')
        plt.savefig('images/distributions.png')
        plt.show()