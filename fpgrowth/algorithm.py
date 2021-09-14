from collections import defaultdict
from itertools import chain, combinations
from typing import Dict, List

from fpgrowth.node import Node


class FPGrowth:
    """
    FP Growth algorithm.
    Using the frequent item list, frequency, constructs the FPGrowth tree.
    It then traverses it to recover assocation rules.
    """

    def __init__(
        self,
        item_set_list: List,
        frequency: List,
        support_ratio: float,
        confidence: float,
    ):
        self.item_set_list = item_set_list
        self.frequency = frequency
        self.support_ratio = support_ratio
        self.support = len(item_set_list) * support_ratio
        self.confidence = confidence

        self.header_table: Dict
        self.tree: Node

    def update_header(self, item: Node, target_node: Node, header_table: Dict):
        if header_table[item][1] is None:
            header_table[item][1] = target_node
        else:
            current_node = header_table[item][1]
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = target_node

    def update(self, item: Node, tree_node: Node, header_table: Dict, frequency: float):
        if item in tree_node.children:
            tree_node.children[item].increment(frequency)
        else:
            new_item_node = Node(item, frequency, tree_node)
            tree_node.children[item] = new_item_node
            self.update_header(item, new_item_node, header_table)

        return tree_node.children[item]

    def construct(self, item_set_list: List, frequency: List, min_support: float):
        header_table = defaultdict(int)  # type: ignore
        for idx, item_set in enumerate(item_set_list):
            for item in item_set:
                header_table[item] += frequency[idx]

        header_table = {  # type: ignore
            item: sup for item, sup in header_table.items() if sup >= min_support
        }

        if not header_table:
            return None, None

        for item in header_table:
            header_table[item] = [header_table[item], None]

        fp_tree = Node("Null", 1, None)
        for idx, item_set in enumerate(item_set_list):
            item_set = [item for item in item_set if item in header_table]
            item_set.sort(key=lambda item: header_table[item][0], reverse=True)
            currentNode = fp_tree
            for item in item_set:
                currentNode = self.update(
                    item, currentNode, header_table, frequency[idx]
                )

        return fp_tree, header_table

    def ascend(self, node: Node, prefix_path: List):
        if node.parent is not None:
            prefix_path.append(node.item_name)
            self.ascend(node.parent, prefix_path)

    def find_path(self, base_path: List, header_table: Dict):
        tree_node = header_table[base_path][1]
        cond_paths = []
        frequency = []
        while tree_node is not None:
            prefix_path = []  # type: ignore
            self.ascend(tree_node, prefix_path)
            if len(prefix_path) > 1:
                cond_paths.append(prefix_path[1:])
                frequency.append(tree_node.count)

            tree_node = tree_node.next
        return cond_paths, frequency

    def mine(
        self, header_table: Dict, min_support: float, prefix: List, freq_item_list: List
    ):
        sorted_item_list = [
            item[0]
            for item in sorted(list(header_table.items()), key=lambda p: p[1][0])
        ]
        for item in sorted_item_list:
            new_freq_set = prefix.copy()
            new_freq_set.add(item)  # type: ignore
            freq_item_list.append(new_freq_set)

            conditional_path_base, frequency = self.find_path(item, header_table)
            conditional_tree, new_header_table = self.construct(
                conditional_path_base, frequency, min_support
            )
            if new_header_table is not None:
                self.mine(new_header_table, min_support, new_freq_set, freq_item_list)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

    def calculate_support(self, test_set: List, item_set_list: List):
        count = 0
        for itemSet in item_set_list:
            if set(test_set).issubset(itemSet):
                count += 1
        return count

    def association_rule(
        self, freq_item_set: List, item_set_list: List, min_conf: float
    ):
        rules = []
        for itemSet in freq_item_set:
            subsets = self.powerset(itemSet)
            item_set_support = self.calculate_support(itemSet, item_set_list)
            for s in subsets:
                confidence = float(
                    item_set_support / self.calculate_support(s, item_set_list)
                )
                if confidence > min_conf:
                    rules.append([set(s), set(itemSet.difference(s)), confidence])
        return rules

    def get_frequency_from_list(self, item_set_list: List):
        return [1 for i in range(len(item_set_list))]

    def run(self):
        fp_tree, header_table = self.construct(
            self.item_set_list, self.frequency, self.support_ratio
        )
        if fp_tree is None:
            print("No frequent item set")
        else:
            freq_items = []
            self.mine(header_table, self.support_ratio, set(), freq_items)
            rules = self.association_rule(
                freq_items, self.item_set_list, self.confidence
            )
            return freq_items, rules
