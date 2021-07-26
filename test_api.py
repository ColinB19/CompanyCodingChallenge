import unittest
import json
import numpy as np
import tensorflow as tf

from sentence_encoder_api import get_embeddings
from sentence_encoder_api import app

BASEURL = 'http://localhost:5000/embeddings'

class BasicTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    def tearDown(self):
        pass
    

    def test_general(self):
        # just check that you get a 400 error when
        # trying to access pages that aren't set up.
        response = self.app.get(BASEURL[:-10])

        self.assertEqual(response.status_code, 404)


    @unittest.mock.patch('sentence_encoder_api.get_embeddings')
    def test_single_embeddings(self, get_embeddings):
        # just mocking the model so we don't have to call it.
        get_embeddings.return_value = tf.constant([np.zeros(512)]).numpy()

        # let's test a good response first.
        test_query = "?sentence=This+is+a+good+query."
        response = self.app.get(BASEURL + test_query)
        print(response)
        data = json.loads(response.get_data())

        self.assertEqual(response.status_code, 200)
        # here I'm just making sure I'm parsing the numpy/tensorflow objects correctly
        self.assertEqual(len(list(data.values())[0]), 512)

        # throwing a bad query at it.
        bad_query = "?item=This+is+a+good+query."
        response2 = self.app.get(BASEURL + bad_query)
        self.assertEqual(response2.status_code, 400)


    @unittest.mock.patch('sentence_encoder_api.get_embeddings')
    def test_multiple_embeddings(self, get_embeddings):
        # just mocking the model so we don't have to call it.
        get_embeddings.return_value = tf.constant([np.zeros(512), np.zeros(512)]).numpy()
        # let's test a good response first.
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
        # here I'm just making sure I'm parsing the numpy/tensorflow objects correctly
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
        # just mocking the model so we don't have to call it.
        get_embeddings.return_value = tf.constant([np.ones(512), np.ones(512)]).numpy()
        # let's test a good response first.
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
