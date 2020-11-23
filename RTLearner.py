import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        """
        Node -> [split_label_index, node_val, left_tree_row_idx, right_tree_row_idx ]
        The left tree is the next row, so left_tree_row_idx is node_row_idx + 1
        Tree generation is Depth first
        :param leaf_size:
        :param verbose:
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model = list()

    def author(self):
        return "narora62"

    def add_evidence(self, xdata, ydata):
        """
        :return:
        """
        concat_data = np.concatenate((xdata, ydata[:, None]), axis=1)
        self.model = self.build_tree(concat_data)
        if self.verbose:
            print(f"RTLearner: Verbose True: {self.model}")

    def build_tree(self, data):
        data_rows_len = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]

        # Terminal statement to break recursive callbacks
        if data_rows_len <= self.leaf_size or np.all(y == y[0]):
            # Return the leaf node
            return np.array([["Leaf", np.mean(y), None, None]])

        # Random index
        feature_index = np.random.randint(data.shape[1]-1)

        # Data Column
        data_column = data[:, feature_index]

        # Root node -> median
        splitVal = np.median(data_column)

        if np.array_equal(data[data[:, feature_index] <= splitVal], data):
            return np.array([["Leaf", np.mean(data[:, -1]), "NA", "NA"]], dtype=object)


        # Callback recursive until leaf

        # Left tree -> < median
        left_tree = self.build_tree(data[data_column <= splitVal])

        # Right tree -> > median
        right_tree = self.build_tree(data[data_column > splitVal])

        root = np.array([[feature_index, splitVal, 1, left_tree.shape[0] + 1]])

        return np.concatenate((root, left_tree, right_tree), axis=0)

    def query(self, points):
        Y_out = []
        for data_row in points:
            row = 0
            while True:
                # Idx of feature to split
                feature_index = self.model[row][0]

                if feature_index == "Leaf":
                    Y_out.append(self.model[row][1])
                    break
                else:
                    if data_row[int(feature_index)] <= self.model[row][1]:
                        row += int(self.model[row][2])
                    elif data_row[int(feature_index)] > self.model[row][1]:
                        row += int(self.model[row][3])

        if self.verbose:
            print(f"RTLearner: Output: {Y_out}")

        return np.array(Y_out)