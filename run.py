import argparse
import os
import pandas as pd
import numpy as np
import zipfile
import requests
import sqlite3
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_LINK = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"
DB_PATH = os.path.join(BASE_DIR, 'db.sqlite3')


# Download and extract the dataset
def download_data(download_path):
    print("Downloading Dataset...")
    dataset_path = os.path.join(download_path, 'ml-latest.zip')
    response = requests.get(DATASET_LINK, stream=True)
    with open(dataset_path, 'wb') as handle:
        for data in tqdm(response.iter_content(chunk_size=1024)):
            handle.write(data)

    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)


# Load the dataset into pandas DataFrame
def load_dataset(input_dataset_path):
    if input_dataset_path is not None:
        print("Loading Dataset...")
        movies = pd.read_csv(os.path.join(input_dataset_path, 'movies.csv'))
        movie_ratings = pd.read_csv(os.path.join(input_dataset_path, 'ratings.csv'), usecols=['movieId', 'rating'])
        genome_scores = pd.read_csv(os.path.join(input_dataset_path, 'genome-scores.csv'))
        genome_tags = pd.read_csv(os.path.join(input_dataset_path, 'genome-tags.csv'))
        imdb_links = pd.read_csv(os.path.join(input_dataset_path, 'links.csv'))
        links = imdb_links[['movieId', 'imdbId', 'tmdbId']]
        return genome_scores, genome_tags, movies, movie_ratings, links
    else:
        download_path = str(os.getcwd())
        download_data(download_path)
        return load_dataset(os.path.join(download_path, 'ml-20m'))


# Connect to the SQLite database
def connect_database(database_path):
    if database_path is None:
        database_path = DB_PATH
    return sqlite3.connect(database_path)


# Concatenate tags into a single string
def concatenate_tags(tags):
    return ' '.join(set(tags))

# Calculate ratings
def calculate_ratings(movie_ratings):
    return movie_ratings.groupby('movieId')['rating'].agg(
        rating_mean='mean',
        rating_median='median',
        num_ratings='size'
    ).reset_index()



def calculate_similarity(genome_scores, genome_tags, movies, movie_ratings, chunk_size=500, checkpoint_file="progress_similarity.csv"):
    tf_idf = TfidfVectorizer()

    # Filter relevant tags
    relevant_tags = genome_scores[genome_scores.relevance > 0.3][['movieId', 'tagId']]
    movie_tags = pd.merge(relevant_tags, genome_tags, on='tagId', how='left')[['movieId', 'tagId']]
    movie_tags['tagId'] = movie_tags.tagId.astype(str)
    tags_per_movie = movie_tags.groupby('movieId').agg(
        movie_tags=('tagId', concatenate_tags)
    ).reset_index()

    # Add ratings and merge datasets
    avg_movie_ratings = calculate_ratings(movie_ratings)
    movies_with_ratings = pd.merge(movies, avg_movie_ratings, on='movieId')
    dataset = pd.merge(movies_with_ratings, tags_per_movie, on='movieId', how='left')

    # Split into comparable and non-comparable movies
    dataset.rename(columns={'movieId': 'movie_id'}, inplace=True)
    movies_with_tags = dataset.movie_tags.notnull()
    dataset_with_tags = dataset[movies_with_tags].reset_index(drop=True)
    uncomparable_movies = dataset[~movies_with_tags]

    print("Calculating movie-to-movie similarity...")

    # Vectorize tags and calculate similarity in chunks
    vectorized = tf_idf.fit_transform(dataset_with_tags.movie_tags)
    num_movies = vectorized.shape[0]
    total_chunks = (num_movies + chunk_size - 1) // chunk_size

    processed_chunks = set()
    if os.path.exists(checkpoint_file):
        try:
            processed_chunks = set(pd.read_csv(checkpoint_file, usecols=['chunk'],low_memory=True)['chunk'].unique())
        except Exception as e:
            print(f"Warning: Could not read checkpoint file ({e}). Starting fresh.")

    for chunk_index in tqdm(range(total_chunks), desc="Processing chunks"):
        if chunk_index in processed_chunks:
            continue

        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_movies)

        chunk_vectors = vectorized[chunk_start:chunk_end]
        similarity_chunk = cosine_similarity(chunk_vectors, vectorized)

        chunk_df = [
            (chunk_index, dataset_with_tags.iloc[chunk_start + row_idx]['movie_id'],
             dataset_with_tags.iloc[col_idx]['movie_id'], score)
            for row_idx, row in enumerate(similarity_chunk)
            for col_idx, score in enumerate(row)
        ]

        temp_df = pd.DataFrame(chunk_df, columns=['chunk', 'first_movie_id', 'second_movie_id', 'similarity_score'])
        if os.path.exists(checkpoint_file) and chunk_index not in processed_chunks:
            temp_df.to_csv(checkpoint_file, mode='a', header=False, index=False)
        else:
            temp_df.to_csv(checkpoint_file, index=False)

        time.sleep(0.5)

    # Final processing: Use chunked reading to handle the similarity CSV
    """ similarity_chunks = []
    for chunk in pd.read_csv(checkpoint_file, chunksize=100000):
        similarity_chunks.append(chunk)

    # Combine chunks into a single DataFrame
    movie_similarity_df = pd.concat(similarity_chunks, ignore_index=True) """

    return dataset_with_tags, uncomparable_movies

def write_similarity_to_db(db_connection, csv_file, table_name, chunksize=10000):
    a = 1
    print(f"Writing similarity data to database from {csv_file}...")
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        print('chunk %s' % str(a))
        chunk.drop('chunk',axis='columns',inplace=True)
        chunk.to_sql(table_name, db_connection, if_exists='append', index=False)
        a += 1
    print(f"Finished writing similarity data to the {table_name} table.")



# Write data to SQLite database in chunks
def write_database(df, table_name, db_connection):
    total_length = len(df)
    step = max(1, int(total_length / 100))

    with tqdm(total=total_length) as pbar:
        for i in range(0, total_length, step):
            subset = df.iloc[i: i + step]
            subset.to_sql(table_name, db_connection, if_exists='append', index=False)
            pbar.update(len(subset))


# Fill the database
def fill_database(db_connection, dataset_with_tags, uncomparable_movies, links):
    links.rename(columns={'movieId': 'movie_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)

    dataset_with_tags['comparable'] = True
    uncomparable_movies['comparable'] = False

    print("Writing data to database...")
    write_database(dataset_with_tags, 'recommender_movie', db_connection)
    write_database(uncomparable_movies, 'recommender_movie', db_connection)
    write_database(links, 'recommender_onlinelink', db_connection)
    print("Write Movie Similarity")
    write_similarity_to_db(db_connection,'progress_similarity.csv','recommender_similarity')
    # write_database(movie_to_movie_similarity, 'recommender_similarity', db_connection)
    print("End Write")


def main(dataset_path):
    genome_scores, genome_tags, movies, movie_ratings, links = load_dataset(dataset_path)
    db_connection = connect_database(DB_PATH)
    dataset_with_tags, uncomparable_movies = calculate_similarity(genome_scores, genome_tags, movies, movie_ratings)
    fill_database(db_connection, dataset_with_tags, uncomparable_movies, links)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens dataset & populate Django database.")
    parser.add_argument('-i', '--input-dataset', type=str, help="Path to dataset folder if already downloaded")
    args = parser.parse_args()
    main(args.input_dataset)
