# 6. Delivering personalized movie recommendations with an AI - driven matching system
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
df = pd.read_csv('/content/tamil_100_movies.csv')  # Make sure to upload the correct CSV file

# Build Recommendation System
def recommend_movies(user_input):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    if user_input not in indices:
        print("❌ Movie not found in the list. Please check your spelling or try another movie.")
        return

    idx = indices[user_input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 similar movies

    print(f"\n🎬 You selected: {user_input}")
    print(f"📝 Review: {df['review'].iloc[idx]}")
    print("\n👉 You might also like:")
    for i in sim_scores:
        print(f"- {df['title'].iloc[i[0]]} ({df['genres'].iloc[i[0]]})")

# Run the recommender
if __name__ == "__main__":
    print("🎉 Welcome to the Tamil Movie Recommendation System 🎉\n")
    print("📃 Here are some available movies:\n")
    print(", ".join(df['title'].sample(20).tolist()))  # Show random 10 movies from 100

    while True:
        movie_name = input("\n🔍 Enter a Tamil movie name (exact match): ")
        recommend_movies(movie_name)

        again = input("\n🔁 Do you want another recommendation? (yes/no): ")
        if again.lower() != 'yes':
            print("👋 Thank you for using our AI movie recommender!")
            break