import tensorflow as tf
import pandas as pd
from config import LABELS, FEATURES, TARGET, BATCH_SIZE


class Model:

    def __init__(self):
        self.vars = self._load_data()

    @staticmethod
    def _load_data():
        train = pd.read_csv('data/train.csv', names=FEATURES, header=0)
        test = pd.read_csv('data/test.csv', names=FEATURES, header=0)
        train_y = train.pop(TARGET)
        test_y = test.pop(TARGET)
        return train, test, train_y, test_y

    def train(self):
        train, test, train_y, test_y = self.vars
        my_feature_columns = []
        for key in train.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Build a DNN with 3 hidden layers with 1024, 512, and 256 hidden nodes each.
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=[1024, 512, 256],
            n_classes=len(LABELS)
        )

        classifier.train(
            input_fn=lambda: self._input_fn(train, train_y, training=True),
            steps=5000
        )

        eval_result = classifier.evaluate(input_fn=lambda: self._input_fn(test, test_y, training=False))
        print('\nTest set recall: {accuracy:0.3f}\n'.format(**eval_result))
        return classifier

    def test(self, classifier):
        train, test, train_y, test_y = self.vars
        expected = test_y.to_list()
        predict_x = {
            'YearsCoding': test.YearsCoding.to_list(),
            'Dependents': test.Dependents.to_list(),
            'Age': test.Age.to_list(),
        }

        predictions = classifier.predict(
            input_fn=lambda: self._input_fn_pred(predict_x))

        for pred_dict, exp in zip(predictions, expected):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            prob = 100 * probability
            res = LABELS[class_id]
            print(f'{res}, {prob:0.3f}, {exp}')

    @staticmethod
    def _input_fn(features, labels, training=True, batch_size=BATCH_SIZE):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()
        return dataset.batch(batch_size)

    @staticmethod
    def _input_fn_pred(features, batch_size=BATCH_SIZE):
        """An input function for prediction."""
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

