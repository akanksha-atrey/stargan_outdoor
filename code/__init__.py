import os

DATA_PATH = './data' + os.sep

TRANSIENT_DATA = DATA_PATH + 'transient_attributes'
TRANSIENT_DATA_ANNOTATIONS = DATA_PATH + 'transient_attributes' + os.sep + 'annotations' + os.sep + 'annotations.tsv'
TRANSIENT_DATA_ATTRIBUTES = DATA_PATH + 'transient_attributes' + os.sep + 'annotations' + os.sep + 'attributes.txt'
 
LANDMARK_DATA = DATA_PATH + 'landmarks'
WC_DATA = DATA_PATH + 'world_cities_outdoor'

MODEL_PATH = './resnets'