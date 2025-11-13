import unittest
from source import build_model, load_data

class BasicModelTests(unittest.TestCase):
    def test_build_model_output_shape(self):
        model = build_model()
        # The last Dense layer produces logits for 10 classes
        self.assertEqual(model.output_shape[-1], 10)

    def test_data_shape(self):
        (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(x_train.shape[1:], (32,32,3))
        self.assertEqual(x_test.shape[1:], (32,32,3))

if __name__ == '__main__':
    unittest.main()