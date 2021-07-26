"""
author: Colin Bradley
last updated: July 25th, 2021

description: this app performs unit tests on the sentence_encoder_api.py module. To run it, run `python test_apy.py`.
"""


import unittest
import json
import numpy as np
import tensorflow as tf

from sentence_encoder_api import get_embeddings
from sentence_encoder_api import app

BASEURL = 'http://localhost:5000/embeddings'

class APITest(unittest.TestCase):
    """
    A class used to the sentence_encoder_api.

    ...

    Attributes
    ----------
    app: a test version of the flask app created in sentence_encoder_api.py

    Methods
    -------
    setUp:
        This function sets up and creates a test API to be used in the other methods.
    tearDown:
        No teardown to be done.
    test_general:
        This function makes sure URL's that are not set up cannot be accessed.
    test_single_embeddings:
        This function tests the single sentence functionality of the API. It mocks the result of
        training the tensorflow model, then checks if the correct HTTP codes are produced in several situations
    test_multiple_embeddings:
        Very similar to test_single_embeddings but with multple sentence embeddings.
    test_cosine_similarity:
        Tests the cosine similarity of two sentence embeddings functionality of the API. Again, checking various HTTP codes. 
    """

    def setUp(self):
        """
        This function sets up a testing version of the API. since app is imported directly from sentence_encoder_api.py, no model is trained. 

        Inputs
        ----------
            N/A

        Returns
        ------
            N/A

        """
        self.app = app.test_client()
        self.app.testing = True
    

    def test_general(self):
        """
        This function checks if the 404 error is raised when a user attempts to access a route not set up.  

        Inputs
        ----------
            N/A

        Returns
        ------
            N/A

        """
        # just check for 400 error when trying to access pages that aren't set up.
        response = self.app.get(BASEURL[:-10])

        self.assertEqual(response.status_code, 404)


    @unittest.mock.patch('sentence_encoder_api.get_embeddings')
    def test_single_embeddings(self, get_embeddings):
        """
        This function tests the single_embddings functionality of the API. It mocks the result of the tensorflow model using the @unittest.mock.patch decorator. This let's the functionality of the API to be tested without relying on the external model. It checks for the 200 HTTP code on a successful request, checks the length of the result of the 'model training', and checks for a 400 HTTP code on a bad query.

        Inputs
        ----------
            get_embeddings: mocked result of training the tf model. 

        Returns
        ------
            N/A

        """
        # just mocking the model so it doesn't need to be called. Saves time and errors if the model/website is down
        get_embeddings.return_value = tf.constant([np.zeros(512)]).numpy()

        # test a good response first.
        test_query = "?sentence=This+is+a+good+query."
        response = self.app.get(BASEURL + test_query)

        data = json.loads(response.get_data())

        self.assertEqual(response.status_code, 200)
        # is it parsing the numpy/tensorflow objects correctly?
        self.assertEqual(len(list(data.values())[0]), 512)

        # throwing a bad query at it.
        bad_query = "?item=This+is+a+good+query."
        response2 = self.app.get(BASEURL + bad_query)
        self.assertEqual(response2.status_code, 400)


    @unittest.mock.patch('sentence_encoder_api.get_embeddings')
    def test_multiple_embeddings(self, get_embeddings):
        """
        This function tests the multiple_embddings functionality of the API. It mocks the result of the tensorflow model using the @unittest.mock.patch decorator. This let's the functionality of the API to be tested without relying on the external model. It sends a JSON object to the API then checks for a 200 HTTP code, checks the dimensions of the result of 'model training', then checks for the 405 HTTP code for a GET request (which is not allowed).

        Inputs
        ----------
            get_embeddings: mocked result of training the tf model. 

        Returns
        ------
            N/A

        """
        # just mocking the model so it doesn't need to be called. Saves time and errors if the model/website is down
        get_embeddings.return_value = tf.constant([np.zeros(512), np.zeros(512)]).numpy()
        # test a good response first.
        test_json = json.dumps(
            {
                "sentences": ["the quick brown fox jumped over the lazy dog", "the five boxing wizards jump quickly"]
            }
        )
        response = self.app.post(BASEURL + "/bulk",
                                data=test_json,
                                content_type='application/json')

        data = json.loads(response.get_data())

        self.assertEqual(response.status_code, 200)
        # is it parsing the numpy/tensorflow objects correctly?
        self.assertEqual(len(list(data.values())[0]), 2)
        self.assertEqual(len(list(data.values())[0][0]), 512)            
        self.assertEqual(len(list(data.values())[0][1]), 512)

        # flag a get request
        bad_response_get = self.app.get(BASEURL + "/bulk",
                                data=test_json,
                                content_type='application/json')
        self.assertEqual(bad_response_get.status_code, 405)


    @unittest.mock.patch('sentence_encoder_api.get_embeddings')
    def test_similarity(self, get_embeddings):
        """
        This function tests the cosine_similarity functionality of the API. It mocks the result of the tensorflow model using the @unittest.mock.patch decorator. This let's the functionality of the API to be tested without relying on the external model. It sends a JSON object to the API then checks for a 200 HTTP code, checks the dimensions of the result of 'model training'. It also checks that the cosine similarity between two identical vectors is 1. Finally, it checks for the 405 HTTP code for a GET request (which is not allowed).

        Inputs
        ----------
            get_embeddings: mocked result of training the tf model. 

        Returns
        ------
            N/A

        """
        # just mocking the model so it doesn't need to be called. Saves time and errors if the model/website is down
        get_embeddings.return_value = tf.constant([np.ones(512), np.ones(512)]).numpy()
        # test a good response first.
        test_json = json.dumps(
            {
                "sentence_1": "the quick brown fox jumped over the lazy dog", 
                "sentence_2": "the five boxing wizards jump quickly"
            }
        )
        response = self.app.post(BASEURL + "/similarity",
                                data=test_json,
                                content_type='application/json')

        data = json.loads(response.get_data())

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(float(list(data.values())[0]), 1.0)

        # flag a get request
        bad_response_get = self.app.get(BASEURL + "/similarity",
                                data=test_json,
                                content_type='application/json')
        self.assertEqual(bad_response_get.status_code, 405)


if __name__ == '__main__':
    unittest.main()
