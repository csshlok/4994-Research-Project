import pandas as pd

reviews = pd.read_parquet("C:/Users/csshl/Desktop/4994-Scrapper/features_exctract/combined_reviews.parquet")

ids = [
    96280017, 97185046, 97396427, 98991913, 93934123,
    96619101, 92253631, 95300500, 96748214, 97255811
]

cols = ["review_id", "company_id", "text_norm"]
out = reviews[reviews["review_id"].isin(ids)][cols]
out.to_csv("zero_goal_examples.csv", index=False)
