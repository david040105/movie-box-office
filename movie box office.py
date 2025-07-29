import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------
# STEP 1: Check if movies.csv exists
# -------------------------------
if not os.path.exists("movies.csv"):
    print("⚠️ 'movies.csv' not found. Creating sample dataset...")

    sample_data = {
        'title': ['Avatar', 'Titanic', 'The Avengers', 'Jurassic World', 'Frozen II'],
        'budget': [237000000, 200000000, 220000000, 150000000, 150000000],
        'revenue': [2787965087, 2187463944, 1518812988, 1671713208, 1450026933],
        'release_date': ['2009-12-10', '1997-12-19', '2012-05-04', '2015-06-12', '2019-11-22'],
        'genres': ['Action,Adventure,Fantasy', 'Drama,Romance', 'Action,Sci-Fi', 'Action,Adventure,Sci-Fi', 'Animation,Family'],
        'popularity': [150.4, 100.1, 160.2, 120.5, 98.7],
        'vote_average': [7.9, 7.8, 8.0, 7.1, 7.4]
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv("movies.csv", index=False)
    print("✅ Sample dataset 'movies.csv' created.")

# -------------------------------
# STEP 2: Load the dataset
# -------------------------------
df = pd.read_csv("movies.csv")
print("\n✅ Dataset Loaded:")
print(df.head())

# -------------------------------
# STEP 3: Data Cleaning
# -------------------------------
df = df.dropna(subset=['budget', 'revenue'])
df = df[df['budget'] > 0]
df = df[df['revenue'] > 0]

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

# Extract main genre
df['main_genre'] = df['genres'].apply(lambda x: x.split(",")[0] if pd.notnull(x) else 'Unknown')

# -------------------------------
# STEP 4: Exploratory Data Analysis (EDA)
# -------------------------------

# Top 5 highest-grossing movies
top_movies = df .sort_values('revenue', ascending=False).head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x='revenue', y='title', data=top_movies, palette='coolwarm')
plt.title("Top 5 Highest Grossing Movies")
plt.xlabel("Revenue")
plt.ylabel("Movie Title")
plt.tight_layout()
plt.show()

# Budget vs Revenue
plt.figure(figsize=(8, 6))
sns.scatterplot(x='budget', y='revenue', data=df)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# Genre-wise average revenue
genre_revenue = df.groupby('main_genre')['revenue'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
genre_revenue.plot(kind='bar', color='skyblue')
plt.title("Average Revenue by Main Genre")
plt.ylabel("Average Revenue")
plt.xlabel("Genre")
plt.tight_layout()
plt.show()

# -------------------------------
# STEP 5: Correlation Heatmap
# -------------------------------
numeric_df = df[['budget', 'revenue', 'popularity', 'vote_average']]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------------
# STEP 6: Revenue Prediction Model
# -------------------------------
features = ['budget']
if 'popularity' in df.columns:
    features.append('popularity')
if 'vote_average' in df.columns:
    features.append('vote_average')

X = df[features]
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:")
print("R-squared:", round(r2_score(y_test, y_pred), 4))

# Predict revenue for a custom movie
example = pd.DataFrame([[200_000_000, 100.0, 8.0]], columns=features)
predicted = model.predict(example)[0]
print(f"📈 Predicted Revenue for a $200M budget movie: ${predicted:,.2f}")