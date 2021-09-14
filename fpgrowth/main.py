"""
This file was run to generate the association rule.
Since this is a compute intensive process we ran the algorithm
already and then stored the results. These results are then used to make suggestions
to the user
"""
from pathlib import Path

from fpgrowth.algorithm import FPGrowth
from fpgrowth.utils import initialize_from_file

if __name__ == "__main__":
    min_support = 0.01
    min_confidence = 0.1

    item_set_list, frequency = initialize_from_file(Path("../data/ingredients.csv"))
    fp = FPGrowth(item_set_list, frequency, min_support, min_confidence)
    items, rules = fp.run()
