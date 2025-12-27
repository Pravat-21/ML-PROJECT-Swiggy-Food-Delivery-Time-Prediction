import pandas as pd
import requests
from pathlib import Path
import pytest
import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app

class FastAPIPredictTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        root_path = Path(__file__).parent.parent
        data_path = root_path / "data" / "raw" / "train.csv"
        sample_row = pd.read_csv(data_path).dropna().sample(1)
        cls.data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("hello, welcome to swiggy time prediciton page", response.text.lower())

    def test_predict_success(self):
        response = self.client.post("/predict", json=self.data)
        self.assertEqual(response.status_code, 200)
        



if __name__ == "__main__":
    unittest.main()
