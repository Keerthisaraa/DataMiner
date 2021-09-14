class Node:
    """
    The data structure used to represent the FP Growth Tree structure
    """

    def __init__(self, item_name, frequency, parent):
        self.item_name = item_name
        self.count = frequency
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, frequency):
        """
        Increment the node and frequency

        Args:
            frequency ([int]): frequency of the node in the path
        """
        self.count += frequency

    def display(self, ind=1):
        """
        Print the tree structure

        Args:
            ind (int, optional): [description]. Defaults to 1.
        """
        print("  " * ind, self.item_name, " ", self.count)
        for child in list(self.children.values()):
            child.display(ind + 1)
