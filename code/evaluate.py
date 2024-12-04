import json
import numpy as np
import evaluate


def run_rationale(prediction_path, reference_path):

    rouge = evaluate.load("")
    bleu = evaluate.load("")
    bertscore = evaluate.load("")


    with open(prediction_path, 'r', encoding='utf8') as f_pred:
        predictions_data = json.load(f_pred)

    with open(reference_path, 'r', encoding='utf8') as f_ref:
        references_data = json.load(f_ref)

    predictions = []
    references = []

    for pred, ref in zip(predictions_data, references_data):

        pred_id = str(int(pred['id'])).strip()
        ref_id = str(int(ref['id'])).strip()




    if len(predictions) == 0 or len(references) == 0:
        print("No valid predictions or references to evaluate. Exiting...")
        return


    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references, max_order=4)
    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")

    print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")
    print(f"BLEU: {bleu_results['bleu'] * 100:.2f}")
    print(f"BERTScore: {np.mean(bertscore_results['f1']) * 100:.2f}")



prediction_path = "/Users/ZYY/Desktop/work/ok/qwenmax_en.json"
reference_path = "/Users/ZYY/Desktop/work/data_answer/enanswer.json"

# 运行评估
run_rationale(prediction_path, reference_path)