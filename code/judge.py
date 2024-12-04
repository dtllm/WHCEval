import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Hallucination detection function with ID matching
def run_detect(prediction_path, ground_truth_path):
    # Load prediction and ground truth data
    predictions = json.load(open(prediction_path, 'r', encoding='utf8'))
    ground_truths = json.load(open(ground_truth_path, 'r', encoding='utf8'))

    preds = []
    labels = []
    err_cnt = 0

    # Check if prediction and ground truth lengths are the same
    # if len(predictions) != len(ground_truths):
    #    print("Prediction and ground truth files have different lengths, please check the data files.")
    #   return

    # Create dictionaries to access data by ID for better matching
    pred_dict = {item['id']: item['judgement'] for item in predictions}
    gt_dict = {item['id']: item['judgement'] for item in ground_truths}  # Changed 'label' to 'judgement'

    # Iterate over prediction IDs and compare only matching ones
    for pred_id, judgement in pred_dict.items():
        if pred_id in gt_dict:
            # Process predictions based on 'Yes'/'No' or numeric values
            if isinstance(judgement, str):
                if judgement.startswith('Yes'):
                    preds.append(1)
                elif judgement.startswith('No'):
                    preds.append(0)
                else:
                    err_cnt += 1
                    continue
            else:
                preds.append(judgement)

            # Process ground truth based on 'Yes'/'No' or numeric values
            label = gt_dict[pred_id]
            if isinstance(label, str):
                if label.startswith('Yes'):
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(label)

    # Calculate evaluation metrics
    hallu_p = precision_score(labels, preds, pos_label=1, average='binary')
    hallu_r = recall_score(labels, preds, pos_label=1, average='binary')
    hallu_f1 = f1_score(labels, preds, pos_label=1, average='binary')

    non_hallu_p = precision_score(labels, preds, pos_label=0, average='binary')
    non_hallu_r = recall_score(labels, preds, pos_label=0, average='binary')
    non_hallu_f1 = f1_score(labels, preds, pos_label=0, average='binary')

    avg_acc = accuracy_score(labels, preds)
    avg_p = (hallu_p + non_hallu_p) / 2
    avg_r = (hallu_r + non_hallu_r) / 2
    macro_f1 = f1_score(labels, preds, average='macro')

    # Output evaluation results
    print(f"hallu_p: {hallu_p * 100:.2f}\thallu_r: {hallu_r * 100:.2f}\thallu_f1: {hallu_f1 * 100:.2f}\t"
          f"non_hallu_p: {non_hallu_p * 100:.2f}\tnon_hallu_r: {non_hallu_r * 100:.2f}\tnon_hallu_f1: {non_hallu_f1 * 100:.2f}\t"
          f"avg_acc: {avg_acc * 100:.2f}\tavg_p: {avg_p * 100:.2f}\tavg_r: {avg_r * 100:.2f}\tmacro_f1: {macro_f1 * 100:.2f}")


# Specify the prediction and ground truth file paths
prediction_path = ""
ground_truth_path = ""

# Run hallucination detection with ID matching
run_detect(prediction_path, ground_truth_path)