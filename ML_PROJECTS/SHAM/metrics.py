"""
  File: metrics.py

  Description:
    Machine-Learning metric calculations

"""

def gen_model_metrics(l_known_values, l_predicted_values):
    """
     Iterate over known-values and corresponding predicted-values to
     calculate totals for TP,FP,TN,FN. From these, calculate Accuracy, Precision, Recall, 
     and F1-Score.
     Required args:
      * l_known_values (list): known label values
      * l_predicted_values (list): values predicted by (trained) model
     Return (dict): all calcuated metric values 
    """
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    N_messages = len(l_known_values)
    for i in range(N_messages):
        true_pos += int(l_known_values[i] == 1 and l_predicted_values[i] == 1)
        true_neg += int(l_known_values[i] == 0 and l_predicted_values[i] == 0)
        false_pos += int(l_known_values[i] == 0 and l_predicted_values[i] == 1)
        false_neg += int(l_known_values[i] == 1 and l_predicted_values[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1score = 2 * precision * recall / (precision + recall)

    d_metrics = {
      'total': N_messages,
      'TP': true_pos,
      'TN': true_neg,
      'FP': false_pos,
      'FN': false_neg,
      'precision': precision,
      'recall': recall,
      'f1score': f1score,
      'accuracy': accuracy
    }

    return(d_metrics)
