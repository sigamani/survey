from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['YearsCoding','YearsCodingProf', 'Dependents', 'Age']

train = pd.read_csv('data/train.csv', names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv('data/test.csv', names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('YearsCodingProf')
test_y = test.pop('YearsCodingProf')



def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)



# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # The model must choose between 11 classes.
    hidden_units=[1024, 512, 256],
    #n_classes=4
    n_classes=4
    )

# Train the Model.
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=10000
    )

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
