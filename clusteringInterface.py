from abc import ABC, abstractmethod

class ClusteringInterface(ABC):

    @abstractmethod
    def cluster(self):
        pass

    @staticmethod
    def divide_and_average(input_list, parts):
        # input_list = [x for x in input_list if x != 0]
        total_numbers = len(input_list)
        average_size = total_numbers // parts
        remaining = total_numbers % parts

        result = []

        start = 0
        for i in range(parts):
            size = average_size + (1 if i < remaining else 0)
            end = start + size
            sublist = input_list[start:end]
            try:
                average = sum(sublist) / len(sublist)
            except ZeroDivisionError:
                average = 0
            result.append(average)
            start = end

        return result

    @staticmethod
    def divide_operations(input_list, parts):

        # Initialize an empty dictionary to hold the sums
        sum_all = {}
        # Iterate over each dictionary in the list
        for d in input_list:
            # Iterate over each key, value pair in the dictionary
            for key, value in d.items():
                # Add the value to the corresponding key in the sum_dict
                if key in sum_all:
                    sum_all[key] += value
                else:
                    sum_all[key] = value



        # input_list = [x for x in input_list if x != 0]
        total_numbers = len(input_list)
        # print(total_numbers)
        average_size = total_numbers // parts
        remaining = total_numbers % parts

        result = []

        start = 0
        for i in range(parts):
            size = average_size + (1 if i < remaining else 0)
            end = start + size
            sublist = input_list[start:end]

            summed_dict = {}
            for d in sublist:
                for key, value in d.items():
                    if key in summed_dict:
                        summed_dict[key] += value
                    else:
                        summed_dict[key] = value

            result.append(summed_dict)
            start = end

        return [result, sum_all]
