from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from config import LABELS, FEATURES, TARGET, BATCH_SIZE

train = pd.read_csv('data/train.csv', names=FEATURES, header=0)
test = pd.read_csv('data/test.csv', names=FEATURES, header=0)
train_y = train.pop(TARGET)
test_y = test.pop(TARGET)

def input_fn(features, labels, training=True, batch_size=BATCH_SIZE):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def input_fn2(features, batch_size=BATCH_SIZE):
	"""An input function for prediction."""
	return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


class Model:


    def train(self):
        # Feature columns describe how to use the input.
        my_feature_columns = []
        for key in train.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
        # Build a DNN with 3 hidden layers with 1024, 512, and 256hidden nodes each.
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=[1024, 512, 256],
            n_classes=len(LABELS)
            )
        
        # Train the Model.
        classifier.train(
            input_fn=lambda: input_fn(train, train_y, training=True),
            steps=5000
            )
        eval_result = classifier.evaluate(
            input_fn=lambda: input_fn(test, test_y, training=False))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        expected = test_y.to_list()
        predict_x = {
        	'YearsCoding': test.YearsCoding.to_list(),
        	'Dependents': test.Dependents.to_list(),
        	'Age': test.Age.to_list(),
        }
        
        
        predictions = classifier.predict(
        	input_fn=lambda: input_fn2(predict_x))
        
        
        for pred_dict, expec in zip(predictions, expected):
        	class_id = pred_dict['class_ids'][0]
        	probability = pred_dict['probabilities'][class_id]
        
        	print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        		LABELS[class_id], 100 * probability, expec))

