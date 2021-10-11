import pandas as pd
import numpy as np
from ast import literal_eval

import pymongo

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold

from keras.layers import  Input, concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten
from bson.objectid import ObjectId

from difflib import SequenceMatcher

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_top_3(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def transform_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
def concatenate_values(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def get_tmdb_recommendations(count_matrix, data_tm, title):
    given_id = get_ml_movie_with_title(data_tm, title)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    data_tm = data_tm.reset_index()
    given_index = data_tm.index[data_tm['movie_id'] == given_id].tolist()[0]

    all_similar = list(enumerate(cosine_sim[given_index]))
    all_similar = sorted(all_similar , key = lambda x: x[1] , reverse = True)
    most_similar = [movie for movie in all_similar if movie[1] > 0][1:11]
    return data_tm['movie_title'].iloc[[i[0] for i in most_similar]]

def get_surprise_recommendations(algorithm, trainset, uid):
    predictions = []
    for ii in trainset.all_items():
        ii = trainset.to_raw_iid(ii)
        predictions.append(algorithm.predict(uid, ii, verbose = False))

    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    return predictions

def get_keras_recommendations(model, data_train_x, uid):
    predictions = list(enumerate(model.predict([[np.asarray([uid]*len(data_train_x)), data_train_x[:,1]]]).flatten().tolist()))
    predictions = sorted(predictions , key = lambda x: x[1] , reverse = True)
    return data_train_x[:,1][[i[0] for i in predictions]]

def read_tmdb_dataset():
    df1=pd.read_csv('./tmdb_dataset/tmdb_5000_credits.csv')
    df2=pd.read_csv('./tmdb_dataset/tmdb_5000_movies.csv')
    df1.columns = ['movie_id','movie_title','cast','crew']
    df2.columns = ['budget','genres','homepage','movie_id','keywords','original_language','original_title','overview','popularity','production_companies','production_countries','release_date','revenue','runtime','spoken_languages','status','tagline','title','vote_average','vote_count']

    data_tm = df2.merge(df1,on='movie_id')
    data_tm.head(5)

    for attr in ['cast', 'crew', 'keywords', 'genres']:
        data_tm[attr] = data_tm[attr].apply(literal_eval)
        
    data_tm['director'] = data_tm['crew'].apply(get_director)
        
    for attr in ['cast', 'keywords', 'genres']:
        data_tm[attr] = data_tm[attr].apply(get_top_3)
        
    for attr in ['cast', 'keywords', 'director', 'genres']:
        data_tm[attr] = data_tm[attr].apply(transform_data)
        
    data_tm['overview'] = data_tm.apply(concatenate_values, axis=1)

    return data_tm
    
def read_movielens_dataset():
    movies = pd.read_csv('./movielens_dataset/u.item' , header = None , sep = "|" , encoding='latin-1')
    movies.columns = ['movie_id' , 'movie_title' , 'release_date' , 'video_release_date' ,
                'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
                'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
                'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
                'Thriller' , 'War' , 'Western']

    users = pd.read_csv('./movielens_dataset/u.user', header= None , sep = '|')
    users.columns = ['user_id' , 'age' , 'gender' , 'occupation' , 'zip_code']

    client = pymongo.MongoClient("mongodb+srv://cinema:cinema@cinemadb.emkyl.mongodb.net/cinema?retryWrites=true&w=majority")
    db = client.cinemadb

    user_objects = [i for i in db.users.find()]
    users_db = pd.DataFrame([{'user_id': i['rownum'] + len(users), 'age': 25, 'gender': 'M', 'occupation': 'occupation', 'zip_code': '21000'} for i in user_objects if db.reviews.find({'user_id': ObjectId(i['_id'])}).count() > 0])
    ratings_db = pd.DataFrame([{'user_id': list(filter(lambda u: u['_id'] == i['user_id'], user_objects))[0]['rownum'] + len(users), 'movie_id': int(i['movie_id']), 'rating': 5, 'timestamp': 0} for i in db.reviews.find()])
    
    all_users = pd.concat([users, users_db])
    all_users.reset_index()

    ratings = pd.read_csv('./movielens_dataset/u.data', header= None , sep = '\t')
    ratings.columns = ['user_id' , 'movie_id' , 'rating' , 'timestamp']


    all_ratings = pd.concat([ratings, ratings_db])
    all_ratings.reset_index()

    data_ml = all_ratings.merge(all_users , on='user_id')
    data_ml = data_ml.merge(movies , on='movie_id')

    return data_ml, movies, len(users)


def get_ml_movie_with_title(movies, title):
    return movies[movies['movie_title'] == sorted(
        movies['movie_title'], 
        reverse=True, 
        key=lambda x: SequenceMatcher(None, x, title).ratio()
    )[0]]['movie_id'].to_numpy()[0]
    
def train_surprise_model(data_ml):
    reader = Reader()
    dataset_surprise = Dataset.load_from_df(data_ml[['user_id', 'movie_id', 'rating']], reader)
    kf = KFold(n_splits=5)
    kf.split(dataset_surprise)
    trainset = dataset_surprise.build_full_trainset()
    algorithm = SVD(n_factors = 200 , lr_all = 0.005 , reg_all = 0.02 , n_epochs = 40 , init_std_dev = 0.05)
    algorithm.fit(trainset)
    return algorithm, trainset

def train_keras_model(data_ml):
    data_keras = data_ml.sample(frac = 1)
    data_train_x = np.array(data_keras[['user_id' , 'movie_id']].values)
    data_train_y = np.array(data_keras['rating'].values)
    x_train, x_test, y_train, y_test = train_test_split(data_train_x, data_train_y, test_size=0.2, random_state=98)
    n_factors = 50
    n_users = len(np.unique(data_keras['user_id']))
    n_movies = len(np.unique(data_keras['movie_id']))

    user_input = Input(shape=(1,))
    user_embeddings = Embedding(input_dim = n_users+1, output_dim=n_factors, input_length=1)(user_input)
    user_vector = Flatten()(user_embeddings)

    movie_input = Input(shape = (1,))
    movie_embeddings = Embedding(input_dim = n_movies+1 , output_dim = n_factors , input_length = 1)(movie_input)
    movie_vector = Flatten()(movie_embeddings)

    merged_vectors = concatenate([user_vector, movie_vector])
    dense_layer_1 = Dense(100 , activation = 'relu')(merged_vectors)
    dense_layer_3 = Dropout(.5)(dense_layer_1)
    dense_layer_2 = Dense(1)(dense_layer_3)
    model = Model([user_input, movie_input], dense_layer_2)
    model.compile(loss='mean_squared_error', optimizer='adam' ,metrics = ['accuracy'] )
    model.fit(
        x = [x_train[:,0] , x_train[:,1]] , 
        y =y_train , batch_size = 128 , 
        epochs = 20 , 
        validation_data = ([x_test[:,0] , x_test[:,1]] , y_test)
    )

    return model, data_train_x

def train_content_vectorizer(data_tm):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(data_tm['overview'])
    return count_matrix
