# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
import pickle
from PIL import Image
import streamlit as st

# . Load trained Pipeline
model = load_model('Model-final')
