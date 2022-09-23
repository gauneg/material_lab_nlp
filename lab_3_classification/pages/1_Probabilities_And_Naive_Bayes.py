import streamlit as st
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
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

@st.cache
def get_dataset():
    examples = [
        ['today is a good day',"positive"],
        ['it is good news all around',"positive"],
        ['good food good location great day',"positive"],
        ['not bad but great',"positive"],
        ['an excellent opportunity',"positive"],
        ['what a great day',"positive"],
        ['good day',"positive"],
        ['what a bad day today',"negative"],
        ['news bad all around',"negative"],
        ['bad food bad location but it is an',"negative"],
        ['not excellent but bad',"negative"],
        ['not a good opportunity',"negative"],
        ['not great',"negative"]
    ]
    return pd.DataFrame(examples, columns=['Text', 'Labels'])         


df_dset = get_dataset()
# shuf_df = df_dset.sample(frac=1).reset_index()

"""
# Probability And Naive Bias

## Calculation without assumptions

$$ P(Positive=True|F_1 \\cap F_2 \\cap F_3) = \\frac{P(F_1 \\cap F_2 \\cap F_3|Pos) \\times P(Pos)}{P(F_1 \\cap F_2 \\cap F_3)}$$

## ASSUMPTIONS

Assumption Of Independence 

$$P(F_1 \\cap F_2 \\cap F_3 | Pos) = P(F_1|Pos) P(F_2|Pos) P(F_3|Pos)$$
"""

"""
### Dataset
"""
show_dataset = st.checkbox('See Dataset', value=True, key='view_ds')
if show_dataset:
    st.table(df_dset)


"""
## Calculation Of Prior Probabilities
"""
prior_prob = st.checkbox('Prior Probabilities', value=True, key='prior_prob')
pos_examples = df_dset[df_dset['Labels']=='positive'].iloc[:,0].values
neg_examples = df_dset[df_dset['Labels']=='negative'].iloc[:,0].values
pos_dict, conditional_prob_pos = get_wc_dict(pos_examples)
neg_dict, conditional_prob_neg = get_wc_dict(neg_examples)
pos_prior = len(pos_examples)/df_dset.shape[0]
neg_prior = len(neg_examples)/df_dset.shape[0]

if prior_prob:
    col1, col2 = st.columns(2)

    with col1:
        """
        ### POSITIVE CLASS
        """
        show_pos = st.checkbox('Show Table', value=True, key='p-tab')
        if show_pos:
            st.table(pos_examples)
        display_str_pos = '''Prob(pos) = P\\bigg(\\frac{%s}{%s}\\bigg) = \\frac{%s}{%s}''' % ('pos', 'total', len(pos_examples),df_dset.shape[0])

        st.latex(display_str_pos)
    with col2:
        """
        ### NEGATIVE CLASS
        """

        show_neg = st.checkbox('Show Table', value=True, key='n-tab')
        if show_neg:
            st.table(neg_examples)
        display_str_neg = '''Prob(neg) = P\\bigg(\\frac{%s}{%s}\\bigg) = \\frac{%s}{%s}''' % ('neg', 'total', len(neg_examples),df_dset.shape[0])

        st.latex(display_str_neg)

condition_prob = st.checkbox('Conditional Probability ', value=True, key='cond_prob')
if condition_prob:
    col11, col12 = st.columns(2)
    pos_keys = [k for k in pos_dict.keys() if k!='total']
    tab_pos = pd.DataFrame({
        'word': pos_keys,
        'formula_pos': ["p(%s|positive)" % key for key in pos_keys],
        'count_pos':[pos_dict[key] for key in pos_keys],
        'probabilty_pos': [conditional_prob_pos[key] for key in pos_keys],
        'formula_neg': ["p(%s|negative)" % key for key in pos_keys],
        'count_neg':[neg_dict[key] for key in pos_keys],
        'probabilty_neg': [conditional_prob_neg[key] for key in pos_keys]
    })
    
    st.table(tab_pos)
    
vocab = list(conditional_prob_neg.keys())

st.text_area("VOCAB", ", ".join(vocab),disabled=True)


input_text = st.text_input("Input(FROM VOCAB ONLY)")

input_tokens = tokenizer.tokenize(input_text)

pos_prob = '''p(pos)'''
neg_prob = '''p(neg)'''
pos_prob_value = '''%s''' % round(pos_prior,2)
neg_prob_value = '''%s''' % round(neg_prior,2)
pos_prob_all = pos_prior
neg_prob_all = neg_prior

for tok in input_tokens:
    if tok not in vocab:
        st.error('%s NOT IN VOCAB' % tok)
    pos_prob +=''' \\times p(%s|positive)''' % tok
    pos_prob_value += ''' * %s''' % round(conditional_prob_pos[tok],3)
    pos_prob_all *= conditional_prob_pos[tok]

    neg_prob +=''' \\times p(%s|negative)''' % tok
    neg_prob_value += ''' * %s''' % round(conditional_prob_neg[tok],3)
    neg_prob_all *= conditional_prob_neg[tok]

st.text("For Positive class:")
st.latex(pos_prob)
st.latex('''=>'''+pos_prob_value +'''='''+str(pos_prob_all))

st.text("For Negative class:")
st.latex(neg_prob)
st.latex('''=>'''+neg_prob_value +'''='''+str(neg_prob_all))



    
            

    











