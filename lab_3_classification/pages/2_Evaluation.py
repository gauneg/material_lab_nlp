from distutils.archive_util import make_archive
from xmlrpc.client import boolean
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
def get_percentage_correct_pred(inp_arr, percent_corr, cat=2):
    y_pred = np.zeros(shape=(10,), dtype=np.bool8)
    if cat==0:
        return y_pred
    if cat==1:
        return np.ones(shape=(10,), dtype=np.bool8)

    for i in range(0, len(inp_arr)):
        if i < percent_corr//10:
            y_pred[i]=inp_arr[i]
        else:
            if y_pred[i] == inp_arr[i]:
                y_pred[i] = not y_pred[i]
    return y_pred
    

def get_tp_fp_tn_fn(y_true, y_pred):
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return tp, fp, tn, fn


"""
# EVALUATION OF CLASSIFICATION
"""

y_true = np.array([False, False, False, True, False, True, True, False, True, True])



corr_pred = st.slider('Correctly Predicted', 0, max_value=100, step=10)


y_pred = get_percentage_correct_pred(y_true, corr_pred)




tp, fp, tn, fn = get_tp_fp_tn_fn(y_true, y_pred)
data = pd.DataFrame({
    'true_labels': y_true,
    'pred_labels': y_pred

})


st.table(data)



show_confusion = st.checkbox('SHOW CONFUSION MATRIX', value=True, key='confusion')

if show_confusion:
    confusion_matrix_arrangement = pd.DataFrame([["T.P",'F.N'], ['F.P', "T.N"]])
    confusion_matrix = pd.DataFrame([[tp, fn], [fp, tn]])

    con_col1, con_col2 = st.columns(2)



    with con_col1:
        st.table(confusion_matrix_arrangement)

    with con_col2:
        st.table(confusion_matrix)

metric_sec = st.checkbox('SHOW METRICS', value=True, key='metrics')

precision = '''precision = \\frac{TP}{TP+FP}=>\\frac{%s}{%s+%s}=>%s''' % (tp, tp, fp, tp/(tp+fp))

recall =  '''recall = \\frac{TP}{TP+FN}=>\\frac{%s}{%s+%s}=>%s''' % (tp, tp, fn, tp/(tp+fn))


f1_score_form = ''' F_{1} = \\frac{2}{\\frac{1}{recall}+\\frac{1}{precision}} = %s''' % f1_score(y_true, y_pred)
# =>\\frac{1}{\\frac{1}{%s}+\\frac{1}{%s}}=>%s''' #% (precision, recall, (2/(1/precision)*(1/recall)))
st.latex(precision)

st.latex(recall)

st.latex(f1_score_form)
