from symbol import tfpdef
import streamlit as st
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
tokenizer = TreebankWordTokenizer()

def get_wc_dict(sentence_arr):
    ## add 1 to each word count to avoid it from becoming zero, it takes care if it's absent from either vocabulary (alpha=1)
    count_dict = {'total': 0}
    for word in [tok for ex in sentence_arr for tok in tokenizer.tokenize(ex)]:
        if word not in count_dict.keys():
            count_dict[word] = 0
        count_dict[word] += 1 
        count_dict['total'] += 1
    prob_dict = {k:v/count_dict['total'] for k,v in count_dict.items() if k!='total'}
    return count_dict, prob_dict


"""
# Feature Extraction
"""
examples = [
        'today is a good day',
        'it is good news all around',
        'good food good location great day',
        'not bad but great',
        'an excellent opportunity',
        'what a great day',
        'good day',
        'what a bad day today',
        'news bad all around',
        'bad food bad location but it is an',
        'not excellent but bad',
        'not a good opportunity',
        'not great'
    ]

df = pd.DataFrame({"TEXT": examples})

count_dict, _ = get_wc_dict(examples)
count_vectoriser = CountVectorizer()
cnt_feat = count_vectoriser.fit_transform(examples)
tfidf_trans = TfidfTransformer()
tfidf_trans.fit(cnt_feat)

df_vocab = pd.DataFrame({
    'words': [word for word in count_dict.keys() if word!='total'],
    'counts': [count_dict[word] for word in count_dict.keys() if word!='total']
})

col1, col2 = st.columns(2)

with col1:
    """
    ### INPUT DATA
    """
    st.table(df)

with col2:
    """
    ### Vocab
    """
    st.table(df_vocab)


txt_inputs = st.text_input(label='Input String')
features = count_vectoriser.transform([txt_inputs])
tf_idf_features = tfidf_trans.transform(features)
features = features.toarray().tolist()
tf_idf_features = tf_idf_features.toarray().tolist()
columns = count_vectoriser.get_feature_names()
values = [txt_inputs]+ features[0]
values2 = [txt_inputs]+ tf_idf_features[0]
col_headers = ['text'] + columns


"""
### Bag Of Words Features
"""
df = pd.DataFrame([values,values2], columns=col_headers)


st.table(df)

"""
### TF IDF

$df(t)$: Document Frequency of t, the number of documents in the document set that contain term t. 

n: Total number of documents 

$idf(t) = log\\big[\\frac{n}{df(t)}\\big]+1$

To prevent from zero division 1 is added to both numerator and denomenator

$idf(t) = log\\big[\\frac{n+1}{df(t)+1}\\big]+1$

$$\\text{tf-idf(t,d)} = tf(t, d) \\times idf(t)$$

Each row is then normalized to the Eucleadian norm.

$$v_{norm} = \\frac{v}{||v||_2} = \\frac{v}{v_1^2+v_2^2+v_3^2 ... v_n^2}$$

"""


    








