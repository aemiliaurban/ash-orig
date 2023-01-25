import csv
import math

# def csv_data_reader(path: str):
#     with open(path) as f:
#         return f.read().splitlines()


def csv_order_data_reader(path: str):
    data = []
    with open(path, newline="") as f:
        for line in f.readlines():
            if line != '"x"\n':
                data.append(float(line) - 1)
    return data


def csv_merge_data_reader(path: str):
    data = []
    with open(path, newline="") as f:
        for line in f.readlines():
            if line != '"V1","V2"\n':
                data.append([int(x) for x in line.split(",")])
    return data


class RDataParser:
    def __init__(self, input_data):
        self.merge_matrix = [map(float, x) for x in input_data["merge_matrix"]]
        self.joining_height = [float(x) for x in input_data["joining_height"]]
        self.order = [float(x) for x in input_data["order"]]
        self.labels = input_data["labels"]
        self.max_tree_height: int = math.ceil(max(self.joining_height))
        self.height_marks: dict = self.create_height_marks()

    def convert_merge_matrix(self):
        transformed_matrix = []
        for node in self.merge_matrix:
            new_node = []
            for el in node:
                if el < 0:
                    transformed_el = abs(el) - 1
                else:
                    transformed_el = el + len(self.merge_matrix)
                new_node.append(transformed_el)
            transformed_matrix.append(new_node)

        self.merge_matrix = transformed_matrix

    def add_joining_height(self):
        # TODO: error for len merge matrix != len joining height
        for index in range(len(self.merge_matrix)):
            self.merge_matrix[index].append(self.joining_height[index])
            self.merge_matrix[index].append(self.order[index])

    def create_height_marks(self) -> dict[int | float, str]:
        height_marks = {}
        for step in range(len(self.joining_height)):
            height_marks[self.joining_height[step]] = f"Formed cluster {str(step+1)}"
        return height_marks
