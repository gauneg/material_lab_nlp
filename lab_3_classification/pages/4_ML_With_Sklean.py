import streamlit as st
import pandas as pd

"""
# STEPS FOR ML

"""

if st.button('STEP-1'):
    """
    ## 1. Understanding the type of problem (Regression/Classification)
    
    Recognising the source of text. *government doc/ mystery work*
    """

if st.button('STEP-2'):
    """

    ## 2. Data Preparation 
    
    1. Splitting data into Train/Test split. `sklearn.model_selection.train_test_split`
    2. Learning to encode the features from text in the training split of the dataset. 
    3. Encoding the text from training spit  dataset i.e. converting from text -> numeric representation.
    4. Encoding the test dataset for evaluation
    why?
    
    1. BoW Representation: CountVectorizer
    2. TF-IDF
    3. Using Word Embedding

    """

if st.button('STEP-3'):
    """
    ## 3. Selecting And Training
    
    1. Select ML model for the job. Based on the type of input/output and the type of problem (classification/regression)
    2. Train (fit) model on your training dataset.

    """

if st.button('STEP-4'):
    """
    ## 4. Evaluation 
    
    1. Selecting evaluation metrics for the task?
    
    *Classification*: (Precision, Recall, F1) 
    
    *Regressive*: (Numerical error MAE, RMSE )

    """

