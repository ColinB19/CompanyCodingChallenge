"""
author: Colin Bradley
last updated: July 25th, 2021

description: I have prepared a quick API according to the steps outline in "Data_Science_Coding_Challenge.pdf". This script handles both the model import from Tensorflow Hub and the API setup using Flask. To run the API, run `python sentence_encoder_api.py`.
"""


from flask import Flask
from flask import request
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)

#######################
######## Model ########
#######################

# global allows me to reference the object in the functions before I've actually assigned it.
global embed
def get_model():
    """ This function retrieves the model from Tensorflow Hub."""
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return model

def get_embeddings(data: list):
    """
    A function that takes in a list of sentences and returns a numpy array of embeddings for each sentence. 
    Each sentence is embedded as a row in the numpy array.
    """

    embeddings = embed(data)
    # we convert the tf tensor to a numpy array for easier handling later
    return embeddings.numpy()

#######################
##### App Routes ######
#######################

@app.route("/embeddings", methods = ['GET'])
def single_embeddings() -> dict:
    """Takes in a query string the user inputs via '...?sentence=...' and returns the encoded sentence."""

    try:
        sentence = request.args['sentence']
    except:
        return "Error: Looks like you didn't input a sentence to encode.\n", 400

    # reason for putting sentence in a list: the tf model takes a list of strings as an argument.
    embedding = get_embeddings([sentence])
    return {"embedding":embedding.tolist()[0]}


@app.route("/embeddings/bulk", methods = ['POST'])
def multiple_embeddings()-> dict:
    """Retrieves an input JSON from a user and returns a JSON of embeddings for each sentence contained in the input."""

    data = request.get_json()
    embeddings = get_embeddings(list(data.values())[0])
    return {"embeddings":embeddings.tolist()}


@app.route("/embeddings/similarity", methods = ['POST'])
def cosine_similarity()-> dict:
    """Retrieves a JSON from a user containing 2 sentences and returns a JSON containing the cosine similarity of the encoded sentences."""

    data = request.get_json()
    
    if len(list(data.values())) != 2:
        # this would do one of two things without this catch.
        # 1. If there is only 1 sentence in the payload, we will get an internal error (500)
        # 2. If there are more than two sentences, no error!
        # I want to throw a bad request error
        return "Error 400: You must send only two sentences, each with their own label.", 400
    
    embeddings = get_embeddings(list(data.values()))

    # cosine similarity is jus the dot product of the normalized vectors
    a = embeddings[0]/np.linalg.norm(embeddings[0])
    b = embeddings[1]/np.linalg.norm(embeddings[1])
    sim = np.dot(a,b)

    return {"similarity":str(round(sim,2))}

#######################
### Error Handling ####
#######################

@app.errorhandler(404)
def not_found(e):
    """This is here so a user cannot access a route not set up."""

    return "Error 404: Not a valid page/command\n", 404


@app.errorhandler(405)
def invalid(e):
    """Invalid method error. I.e. sending a pdf or using spaces in the query string. 
    Could also send a GET request instead of a POST request or visa versa."""

    return "Error 405: Method Not Allowed. Invalid input JSON or invalid request method.\n", 405


@app.errorhandler(400)
def not_supported(e):
    """Bad request handler (a query the API doesn't recognize)"""

    return "Error 400: Bad Request\n", 400


@app.errorhandler(500)
def app_error(e):
    """This is if something happens on the app side."""

    return "Error 500: Something went wrong on this end.", 500


#######################
######### Run #########
#######################

if __name__ == "__main__":
    # tf sentence encoder model loading. Happens before `app.run()` in order to avoid
    # reloading the model every request.
    embed = get_model()
    app.run(host="localhost")