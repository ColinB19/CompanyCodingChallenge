# Data Science Coding Challenge
## by: Colin Bradley
-------------------------------

## Introduction
---------------
This is a fun little project I completed for a Coding Challenge for [OpenTable](https://www.opentable.com/). It is an API designed to use the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) to get sentence embeddings in various formats (single sentences, lists of sentences, and the cosine similarity between two sentence embeddings). I used flask to create the API and pythons build in `unittest` module for testing. Check out the code and let me know if you have any questions!

## Getting started
------------------
1. If you want a detailed look at the packages and dependencies then check out `requirements.txt`. If you just want to get the program running. Go ahead and create a new environment (I used Anaconda for this):
```
conda create --name myenv python=3.9
```
2. Then install necessary packages with pip:
```
pip install numpy flask tensorflow tensorflow_hub
```
3. Run `makejson.py` if you want to make some quick test jsons (the test sentences are the sentences provded in the problem description):
```
python makejson.py
```
4. When you're ready to test out the API, run the `sentence_encoder_api.py` module:
```
python sentence_encoder_api.py
```
5. Once you have the API running and the JSON's made, you can run the test cases found in the problem description:
    1. `curl "http://localhost:5000/embeddings?sentence=the+quick+brown+fox"`
    2. `curl -X POST -H "Content-Type: application/json" -d @payload_2.json http://localhost:5000/embeddings/bulk`
    3. `curl -X POST -H "Content-Type: application/json" -d @payload_3.json http://localhost:5000/embeddings/similarity`

6. Unit testing is done in `test_api.py`. It's simple to run!
```
python test_api.py
```
