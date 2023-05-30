# Data Science Coding Challenge

This is a fun little project I completed as a coding challenge for a job interview. It is an API designed to use the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) to get sentence embeddings in various formats (single sentences, lists of sentences, and the cosine similarity between two sentence embeddings). I used flask to create the API and python's build in `unittest` module for testing. Check out the code and let me know if you have any questions!

## Getting started
1. To get started, clone this repo:
```
git clone https://github.com/ColinB19/CompanyCodingChallenge
```
2. If you want a detailed look at the packages and dependencies then check out `requirements.txt`. If you just want to get the program running. Go ahead and create a new environment (I used Anaconda for this):
```
conda create --name myenv python=3.9
conda activate myenv
```
3. Then install necessary packages with pip:
```
pip install numpy flask tensorflow tensorflow_hub
```
4. Run `makejson.py` if you want to make some quick test JSON's (the test sentences are the sentences provided in the problem description):
```
python makejson.py
```
5. When you're ready to test out the API, run the `sentence_encoder_api.py` module:
```
python sentence_encoder_api.py
```
6. Once you have the API running and the JSON's made, you can run the test cases found in the problem description:
    1. `curl "http://localhost:5000/embeddings?sentence=the+quick+brown+fox"`
    2. `curl -X POST -H "Content-Type: application/json" -d @payload_2.json http://localhost:5000/embeddings/bulk`
    3. `curl -X POST -H "Content-Type: application/json" -d @payload_3.json http://localhost:5000/embeddings/similarity`

7. Unit testing is done in `test_api.py`. It's simple to run!
```
python test_api.py
```
