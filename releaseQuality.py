import json
import pandas as pd
import dataHelper
import statics
class ReleaseQuality:

    def __init__(self):
        # with open('data/' + language + '/' + metric + '/cluster_labels.json') as json_file:
        #     self.clusters = json.load(json_file)

        with open('data/softdtw/patterns_metrics.json') as json_file:
            self.pattens_metrics = json.load(json_file)

        with open('data/softdtw/patterns_switches_metrics.json') as json_file:
            self.pattens_switches_metrics = json.load(json_file)

        self.types = ['architecture', 'design', 'testability', 'implementation', 'test', 'total', 'complexity', 'coupling', 'cohesion']

    def pattens_quality(self):
        """
            Analyzes the quality of patterns based on various metrics, applying the Scott-Knott ESD
            statistical test to detect significant differences in the patterns. The function processes
            pattern metrics, renames clusters, computes statistical ranks, and exports the ranking
            results to a CSV file
        """
        significants = {}

        for metric_type in self.types:
            print(metric_type)
            significants[metric_type] = {}
            # print(metric_type)
            smells = self.pattens_metrics[metric_type]

            renamed_smells = {}
            for key, value in smells.items():
                renamed_smells[dataHelper.rename_cluster(key)] = value
            smells = renamed_smells

            pure_skott_knott = []
            for pattern, values in smells.items():
                for value in values:
                    pure_skott_knott.append([str(pattern), value])

            data = pd.DataFrame.from_dict(smells, orient='index').transpose()
            data = data.apply(lambda x: x.fillna(x.median()), axis=0)

            pandas2ri.activate()
            sk = importr('ScottKnottESD')
            r_sk = sk.sk_esd(data, version='np')

            # Convert IntVector to a Python list
            int_vector_list = list(r_sk[3])

            # Perform arithmetic operation on the list
            column_order = [x - 1 for x in int_vector_list]
            ranks = list(r_sk[1].astype("int"))
            groups = set(list(r_sk[1].astype("int")))

            if len(groups) > 0:
                # significants.append(metric_type)
                significants[metric_type]['group_counts'] = len(groups)
                from collections import Counter
                # significants[metric_type]['counter'] = dict(Counter(groups))
                significants[metric_type]['ratios'] = float(
                    max(dict(Counter(ranks)).values()) / min(dict(Counter(ranks)).values()))

                means = data.mean()
                median = data.median()
                ranking = pd.DataFrame(
                    {
                        "technique": [data.columns[i] for i in column_order],
                        "rank": r_sk[1].astype("int"),
                        "mean": [round(means[i] * 1000, 2) for i in column_order],
                        "median": [round(median[i] * 1000, 2) for i in column_order]
                    }
                )

                print(ranking)
                ranking.to_csv('data/softdtw/' + metric_type + '.csv', index=False)
                print('========')

    def switches_quality(self):
        """
            Analyzes the quality of patterns based on various metrics, applying the Scott-Knott ESD
            statistical test to detect significant differences in the patterns switches. The function processes
            pattern metrics, renames clusters, computes statistical ranks, and exports the ranking
            results to a CSV file
        """
        significants = {}

        for metric_type in self.types:
            print(metric_type)
            significants[metric_type] = {}
            # print(metric_type)
            smells = self.pattens_switches_metrics[metric_type]

            renamed_smells = {}
            for key, value in smells.items():
                renamed_smells[dataHelper.rename_cluster(key)] = value
            smells = renamed_smells

            pure_skott_knott = []
            for pattern, values in smells.items():
                for value in values:
                    pure_skott_knott.append([str(pattern), value])

            data = pd.DataFrame.from_dict(smells, orient='index').transpose()
            data = data.apply(lambda x: x.fillna(x.median()), axis=0)

            pandas2ri.activate()
            sk = importr('ScottKnottESD')
            r_sk = sk.sk_esd(data, version='np')

            # Convert IntVector to a Python list
            int_vector_list = list(r_sk[3])

            # Perform arithmetic operation on the list
            column_order = [x - 1 for x in int_vector_list]
            ranks = list(r_sk[1].astype("int"))
            groups = set(list(r_sk[1].astype("int")))

            if len(groups) > 0:
                # significants.append(metric_type)
                significants[metric_type]['group_counts'] = len(groups)
                from collections import Counter
                # significants[metric_type]['counter'] = dict(Counter(groups))
                significants[metric_type]['ratios'] = float(
                    max(dict(Counter(ranks)).values()) / min(dict(Counter(ranks)).values()))

                means = data.mean()
                median = data.median()
                ranking = pd.DataFrame(
                    {
                        "technique": [data.columns[i] for i in column_order],
                        "rank": r_sk[1].astype("int"),
                        "mean": [round(means[i] * 1000, 2) for i in column_order],
                        "median": [round(median[i] * 1000, 2) for i in column_order]
                    }
                )

                print(ranking)
                ranking.to_csv('data/softdtw/switches_' + metric_type + '.csv', index=False)
                print('========')

