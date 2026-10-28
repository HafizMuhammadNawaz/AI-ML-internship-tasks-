AI/ML Internship - Task: Movie Recommendation System

This project focuses on building a Content-Based Movie Recommendation System using Python, Pandas, Scikit-Learn, and NLTK, leveraging the TMDB 5000 Movies and Credits datasets.

Task Objective

The primary objective is to preprocess raw movie metadata, engineer informative text tags (comprising overviews, genres, keywords, top cast members, and directors), vectorize the text using CountVectorizer, and compute cosine similarity to recommend the top 5 most similar movies based on user selection.

Dataset Overview

Data Source: TMDB 5000 Movies Dataset (tmdb_5000_movies.csv) and TMDB 5000 Credits Dataset (tmdb_5000_credits.csv)

Size: Merged dataset containing 4,806 unique movie entries after cleaning.

Key Features Used: movie_id, title, overview, genres, keywords, cast, and crew.

Workflow & Steps

1. Data Cleaning & Preprocessing

Merging Datasets: Combined movies and credits dataframes on the title column.

Column Filtering: Retained only essential columns (movie_id, title, overview, genres, keywords, cast, crew) and dropped missing values and duplicates.

JSON Parsing: Used ast.literal_eval to extract clean lists of names from stringified dictionary columns for genres, keywords, and cast (limited to the top 3 actors).

Director Extraction: Filtered the crew column to isolate the film director's name.

Space Removal: Standardized multi-word entities (e.g., actor names, genres) by stripping whitespace (e.g., "Sam Worthington" -> "SamWorthington") to ensure the vectorizer treats each entity as a unique token.

2. Feature Engineering (tags Creation)

Tag Aggregation: Combined the tokenized overview, genres, keywords, cast, and crew lists into a single consolidated string column named tags.

Text Stemming: Applied PorterStemmer from NLTK to reduce words to their root forms (e.g., "action", "actions", "acted" -> "action"), reducing feature sparsity.

3. Vectorization & Similarity Computation

Count Vectorization: Utilized Scikit-Learn's CountVectorizer with max_features=5000 and English stop words removal to convert text tags into a numerical bag-of-words matrix of shape (4806, 5000).

Cosine Similarity: Computed pairwise cosine similarity across all vectors to construct a similarity matrix, enabling efficient querying for nearest neighbor movie profiles.

4. Recommendation Engine & Serialization

Recommendation Function: Built a function recommend(movie) that fetches a movie's index, retrieves its similarity scores, sorts them in descending order, and prints the top 5 recommended movie titles.

Model Export: Serialized the processed dataframe and similarity matrix using pickle (movie_list.pkl, similarity.pkl) for future deployment in web applications (e.g., Streamlit).

Key Findings & Output

Model Efficiency: Stemming and feature limitation effectively capture semantic similarities without excessive computational overhead.

Recommendation Quality: Queries yield highly relevant genre, director, and cast-aligned recommendations for tested titles (e.g., Avatar, The Dark Knight Rises).
