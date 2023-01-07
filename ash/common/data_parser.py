class RDataParser:
    def __init__(self, input_data):
        self.merge_matrix = [map(float, x) for x in input_data["merge_matrix"]]
        self.joining_height = [float(x) for x in input_data["joining_height"]]
        self.order = [float(x) for x in input_data["order"]]
        self.labels = input_data["labels"]

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
        print(transformed_matrix)

    def add_joining_height(self):
        # TODO: error for len merge matrix != len joinig height
        for index in range(len(self.merge_matrix)):
            self.merge_matrix[index].append(self.joining_height[index])
            self.merge_matrix[index].append(self.order[index])
        print(self.merge_matrix)
