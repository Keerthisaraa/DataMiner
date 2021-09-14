import pickle
from csv import reader
from pathlib import Path
from typing import List, Optional, Set, Tuple


def initialize_from_file(fname: Path) -> Tuple[List, List]:
    """
    Read the items from the file and create an itemsetlist and frequency

    Args:
        fname (Path): filename

    Returns:
        Tuple[List, List]: Item set list and frequency
    """
    item_set_list = []
    frequency = []

    with open(fname, "r") as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            item_set_list.append(line)
            frequency.append(1)

    return item_set_list, frequency


def sort_suggestions(
    suggestions: List[Tuple[Set[str], float]]
) -> List[Tuple[Set[str], float]]:
    """
    Given a set of suggestions, sort the rules based on the confidence value

    Args:
        suggestions (List[Tuple[Set[str], float]]): List of suggestions

    Returns:
        List[Tuple[Set[str], float]]: List of sorted suggestions
    """
    confidence_list = [suggestion[1] for suggestion in suggestions]
    sort_index = sorted(range(len(confidence_list)), key=lambda k: confidence_list[k])
    # Inverse the sort
    sort_index = sort_index[::-1]
    return [suggestions[i] for i in sort_index]


def remove_duplicates(suggestions):
    existing = {}
    return_suggestions = []
    for suggestion in suggestions:
        suggestion_str = ",".join(suggestion[0])
        if suggestion_str not in existing:
            existing[suggestion_str] = 1
            return_suggestions.append(suggestion)

    return return_suggestions


def make_prediction(
    ingredients: List[str], top_n_suggestions: int, rules_path: str
) -> Optional[List[Tuple[Set[str], float]]]:
    """
    Given a list of ingredients, this function uses the stored association rules
    to fetch the related items as suggestion for the user.
    The results are a set of ingredients and their confidence score in descending order

    Change the rules.pkl file path relative to from where the function is called


    Args:
        ingredients (List[str]): List of ingredients from the user
        top_n_suggestions (int): The number of top suggestions for the ingredient list

    Example usage:
    >>> make_prediction({'garlic_cloves", 'pepper'})

    Returns:
        Optional[Tuple[List[str], float]]: Similar items related to the ingredient list passed if present
        along with the confidence score
    """
    with open(rules_path, "rb") as f:
        rules = pickle.load(f)

    suggestions = [(rule[1], rule[2]) for rule in rules if rule[0] == ingredients]
    sorted_suggestions = sort_suggestions(suggestions)
    return remove_duplicates(sorted_suggestions[:top_n_suggestions])

def read_pickle(path: str) -> List:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data
        