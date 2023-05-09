from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# load the data and create the model
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')
movies['genres'] = movies['genres'].str.replace('|', ' ')

ratings_f = ratings.groupby('userId').filter(lambda x:len(x)>=55)
movie_list_rating = ratings_f.movieId.unique().tolist()

movies = movies[movies.movieId.isin(movie_list_rating)]
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))

mixed = pd.merge(movies, tags, on='movieId', how='left')
mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())

n = 200
svd = TruncatedSVD(n_components=n)
latent_matrix = svd.fit_transform(tfidf_df)

ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on="movieId", how="right")
ratings_f2 = ratings_f1.pivot(index='movieId', columns='userId', values='rating').fillna(0)

svd = TruncatedSVD(n_components=n)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2, index=Final.title.tolist())

latent_matrix_1_df = pd.DataFrame(latent_matrix[:, 0:n], index=Final.title.tolist())

def get_movie_recommendations(title):
    # get the latent factors for the input movie title
    try:
        a_1 = np.array(latent_matrix_1_df.loc[title]).reshape(1, -1)
        a_2 = np.array(latent_matrix_2_df.loc[title]).reshape(1, -1)
    except:
        return []
    
    # compute the cosine similarity between the input movie and all other movies
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)
    
    # compute the hybrid score as the average of the content-based and collaborative filtering scores
    hybrid = ((score_1 + score_2) / 2.0)
    
    # create a DataFrame to store the scores and sort it in descending order
    dictDf = {'content': score_1, 'collaborative': score_2, 'hybrid': hybrid}
    similar = pd.DataFrame(dictDf, index=latent_matrix_1_df.index)
    similar.sort_values('hybrid', ascending=False, inplace=True)
    
    # return the top 10 movie recommendations
    return similar.index[1:11].tolist()

# create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # get input movie from form
    movie = request.form['movie']

    # get the top 10 recommended movies
    recommended_movies = get_movie_recommendations(movie)

    # get the movie information for the recommended movies
    movie_info = []
    for recommended_movie in recommended_movies:
        movie_id = Mapping_file[recommended_movie]
        title = movies.loc[movies['movieId'] == movie_id, 'title'].iloc[0]
        genres = movies.loc[movies['movieId'] == movie_id, 'genres'].iloc[0]
        movie_info.append((title, genres))

    # render the recommendations page with the movie information
    return render_template('recommendations.html', movie_info=movie_info)

if __name__ == '__main__':
    app.run(debug=True)

