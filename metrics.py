import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

#####################################
# metrics reference: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
# NMI: use normalized_mutual_info_score(label, pred, average_method='arithmetic')
# RI: I found adjusted_rand_index is used more often, so use adjusted_rand_score(label, pred)
# Purity, F-Measure: below

def clustering_accuracy(label, pred):
    """referred as purity in stanford webpage""" 
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(pred , np.ndarray):
        pred  = np.array(pred)
        
    K = pred.max() + 1
    C = label.max() + 1
    
    num_correct = 0
    for k in range(K):
        num_same = []
        for c in range(C):
            k_idx = pred == k
            c_idx = label == c
            join = (k_idx * c_idx).sum()
            num_same.append(join)
        num_correct += np.max(num_same)

    accuracy = num_correct / len(label)
    return accuracy

def clustering_fmeasure(label, pred):
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(pred , np.ndarray):
        pred  = np.array(pred)
    
    lpair = label.reshape(-1, 1)
    lpair = lpair == label
    
    ppair = pred.reshape(-1, 1)
    ppair = ppair == pred
    
    lpair = lpair.flatten()
    ppair = ppair.flatten()
    
    f1 = f1_score(lpair, ppair)
    return f1