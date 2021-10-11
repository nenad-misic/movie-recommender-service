import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import pymongo
from bson.objectid import ObjectId

from flask import Flask, request
from app.models import *
from flask_cors import CORS

data_ml, movies, num_of_users = read_movielens_dataset()
data_tm = read_tmdb_dataset()

algorithm, trainset = train_surprise_model(data_ml)
model, data_train_x = train_keras_model(data_ml)
count_matrix = train_content_vectorizer(data_tm)


app = Flask(__name__)
CORS(app)


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.json['user_id']
    
    client = pymongo.MongoClient("mongodb+srv://cinema:cinema@cinemadb.emkyl.mongodb.net/cinema?retryWrites=true&w=majority")
    db = client.cinemadb
    try:
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user is None:
            raise Exception()
    except:
        return {'message': "No user found"}

    rownum = user['rownum']


    reviews = [i for i in db.reviews.find({'user_id': ObjectId(user_id)})]
    if len(reviews) == 0:
        # User has never watched a movie
        return {'content_similar': [], 'surprise_similar': [], 'keras_similar': []}
    else:
        content_similar = []
        for review in reviews:
            movie_title = str(movies[movies['movie_id'] == int(review['movie_id'])]['movie_title'].to_numpy()[0])
            for i in get_tmdb_recommendations(count_matrix, data_tm, movie_title).to_numpy():
                content_similar.append(i)
        content_similar = random.sample(content_similar, min(20, len(content_similar)))

        surprise_similar = get_surprise_recommendations(algorithm, trainset, rownum+num_of_users)[:20]
        surprise_similar = [movies[movies['movie_id'] == e.iid]['movie_title'].to_numpy()[0] for e in surprise_similar]
        
        values = get_keras_recommendations(model, data_train_x, rownum+num_of_users)
        keras_similar = [value for index, value in enumerate(values) if value not in values[ : index]][:20]
        keras_similar = [movies[movies['movie_id'] == e]['movie_title'].to_numpy()[0] for e in keras_similar]
        return {'content_similar': content_similar, 'surprise_similar': surprise_similar, 'keras_similar': keras_similar}

@app.route('/user', methods=['POST'])
def add_user():
    user_name = request.json['name']
    
    client = pymongo.MongoClient("mongodb+srv://cinema:cinema@cinemadb.emkyl.mongodb.net/cinema?retryWrites=true&w=majority")
    db = client.cinemadb
    user_id = db.users.insert_one({'name': user_name, 'rownum': db.users.find().count()+1}).inserted_id
    
    return {"User ID": str(user_id)}

@app.route('/watch', methods=['POST'])
def watch_movie():
    movie_title = request.json['movie_title']
    user_id = request.json['user_id']
    
    client = pymongo.MongoClient("mongodb+srv://cinema:cinema@cinemadb.emkyl.mongodb.net/cinema?retryWrites=true&w=majority")
    db = client.cinemadb
    try:
        if db.users.find_one({"_id": ObjectId(user_id)}) is None:
            raise Exception()
    except:
        return {'message': "No user found"}

    movie_id = get_ml_movie_with_title(movies, movie_title)

    try:
        if db.reviews.find_one({"user_id": ObjectId(user_id), "movie_id": str(movie_id)}) is None:
            raise Exception()
    except:
        db.reviews.insert_one({"user_id": ObjectId(user_id), "movie_id": str(movie_id)})
    
    return {'Message': 'Success'}



@app.route('/models', methods=['GET'])
def refresh_models():
    global data_ml
    global movies
    global data_tm
    global algorithm
    global trainset
    global model
    global data_train_x
    global count_matrix
    global num_of_users

    data_ml, movies, num_of_users = read_movielens_dataset()
    data_tm = read_tmdb_dataset()

    algorithm, trainset = train_surprise_model(data_ml)
    model, data_train_x = train_keras_model(data_ml)
    count_matrix = train_content_vectorizer(data_tm)

    return {'Message': 'Success'}


# predictions = get_surprise_recommendations(algorithm, 1)
# predictions.sort(key=lambda x: x.est, reverse=True)
# for pred in predictions[:10]:
#     print('Surprise: Movie -> {} with Score -> {}'.format(pred.iid , pred.est))


# predictions = get_keras_recommendations(model, shuffled_ids, 1)
# for pred in predictions[:10]:
#     print('Keras: Movie -> {}'.format(pred))

# recommendations = get_tmdb_recommendations(data_tm, 'Hot Fuzz')

# print(recommendations)
