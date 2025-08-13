import pandas as pd
import re
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# -----------------------------------------------
# 1. Load data
# -----------------------------------------------
ratings = pd.read_csv("ratings.csv")
movies  = pd.read_csv("movies.csv")

# Extract year from title
def extract_year(title):
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None

movies["year"] = movies["title"].apply(extract_year)

# -----------------------------------------------
# 2. Helper Functions
# -----------------------------------------------

def get_movies_by_genre_and_year(genre: str, min_year: int = 2000, n: int = 15):
    """Filter movies by genre and year, and return a random sample."""
    pool = movies[
        movies["genres"].str.contains(genre, case=False, na=False)
        & (movies["year"] >= min_year)
    ]
    if pool.empty:
        raise ValueError("No movies match that genre/year combination.")
    return pool.sample(min(n, len(pool)))


def get_user_ratings(movie_sample):
    """Ask the user to rate movies and return as a DataFrame."""
    print("\nPlease rate at least 5 movies (0.5 to 5). Enter 0 to skip.")
    user_ratings = []

    for _, row in movie_sample.iterrows():
        try:
            rating = float(input(f"Rating for '{row['title']}': "))
            if rating >= 0.5:
                user_ratings.append({
                    "userId": 9999,
                    "movieId": row["movieId"],
                    "rating": rating
                })
        except ValueError:
            print("Invalid input. Skipping.")

    return pd.DataFrame(user_ratings)


def append_user_data(existing_ratings, user_ratings):
    """Add new ratings from user to existing ratings DataFrame."""
    return pd.concat([existing_ratings, user_ratings], ignore_index=True)


def train_and_recommend(all_ratings, genre_filter, min_year=2000, user_id=9999, top_n=5):
    """Train a recommender using SVD and recommend top N unseen movies."""
    # Load into Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(all_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Train SVD model
    model = SVD()
    model.fit(trainset)

    # Find movies the user hasn't rated yet
    seen = set(all_ratings[all_ratings['userId'] == user_id]['movieId'])
    unseen_movies = movies[~movies['movieId'].isin(seen)]

    # Filter unseen movies by genre and year
    filtered = unseen_movies[
        (unseen_movies['genres'].str.contains(genre_filter, case=False, na=False)) &
        (unseen_movies['year'] > min_year)
    ]

    if filtered.empty:
        print("❌ No unseen movies found for that genre/year.")
        return []

    # Predict ratings
    predictions = [
        model.predict(str(user_id), str(row['movieId']))
        for _, row in filtered.iterrows()
    ]

    top_predictions = [
    pred for pred in predictions if pred.est > 3
    ]
    top_predictions = sorted(top_predictions, key=lambda x: x.est, reverse=True)[:top_n]

    # Return titles and scores
    results = [
        (movies[movies['movieId'] == int(pred.iid)]['title'].values[0], pred.est)
        for pred in top_predictions
    ]
    return results

# -----------------------------------------------
# 3. Main Program
# -----------------------------------------------

if __name__ == "__main__":
    genre     = input("Enter a genre (e.g., Comedy): ")
    min_year  = int(input("Enter a minimum year (e.g., 2010): "))

    try:
        sample = get_movies_by_genre_and_year(genre, min_year)
    except ValueError as err:
        print(f"\n{err}")
        raise SystemExit

    user_df = get_user_ratings(sample)
    if user_df.empty:
        print("\nNo ratings given — exiting.")
        raise SystemExit

    all_ratings = append_user_data(ratings, user_df)
    recs = train_and_recommend(all_ratings, genre, min_year, top_n=5)

    print("\n🎬  Top recommendations just for you:")
    for title, score in recs:
        print(f"  • {title}  —  predicted rating: {score:.2f}")
