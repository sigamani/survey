import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

""" Labels trained on. These correspond to 0: 0-5 years, 1: 6-17 years, 18-29 years, 30 or more years"""
LABELS = [0, 1, 2, 3]

FEATURES = ['YearsCoding','YearsCodingProf', 'Dependents', 'Age']

TARGET = "YearsCodingProf"

""" Pretrained model """
MODEL_PATH = os.path.join(PROJECT_ROOT, '.', 'model.h5')

BATCH_SIZE = 256 
