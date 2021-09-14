# %%
from utils import make_prediction

make_prediction(ingredients={"onions", 'salt'}, top_n_suggestions=5, rules_path="./rules.pkl")

# %%
