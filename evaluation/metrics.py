"""
@ Author: Omar
"""
import torch
import warnings
from collections import Counter 
from sklearn.metrics import f1_score as F1_sklearn 
import ranx
from pathlib import Path
import os
import re
import string
from datasets import Dataset, DatasetDict

def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def answer_preprocess(answer):
    """Adapted from datasets squad metric. Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(answer.lower())))

 

def perplexity(eval_outputs, output_key='log_probs', metric_prefix=""):    
    targets = []  
    total_loss = 0
    for batch in eval_outputs:        
        labels = batch[ metric_prefix + '_'+'labels']
        targets.extend(labels)
        total_loss+= batch['loss']
    perplexity = torch.exp(total_loss / len(targets))    

    metrics = {}
    metrics['perplexity'] = perplexity.cpu().item()
    return  metrics 

def evaluate_mlm_f1(eval_outputs, output_key='log_probs', metric_prefix=""):    
    predictions = []
    targets = []  
    for batch in eval_outputs:
        log_probs = batch[output_key]
        labels = batch[ metric_prefix + '_'+'labels']
        
        # Get the predicted token IDs for the masked position
        predicted_labels = torch.argmax(log_probs,dim=1)
        predictions.extend(predicted_labels)
        targets.extend(labels)
    
    predictions = torch.stack(predictions).numpy()
    targets = torch.stack(targets).numpy()

    # Calculate the F1 score
    f1 = F1_sklearn(targets, predictions, average="micro")
    metrics = {}
    metrics['f1'] = f1
    return  metrics 


def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
  
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    
    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], 
                            references=[ref_answer])
    
        print("Context: \n", context)
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer

def retrieval(eval_outputs, ignore_index=-100, output_key='log_probs'):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains log_probs and labels for all batches in the evaluation step (either validation or test)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    output_key: str, optional
        Name of the model output in eval_outputs
    """
    metrics = {}    
    mrr, hits_at_1, ignored_predictions, dataset_size = 0, 0, 0, 0   

    for batch in eval_outputs:           
        if output_key in batch:           
            # if type(batch[output_key]) == torch.tensor:
            if torch.is_tensor(batch[output_key]):
               log_probs = batch[output_key].numpy()
            else:                
                return {'MRR@NM':0,'hits@1':0}
        else:
            return {'MRR@NM':0,'hits@1':0}

        labels = batch['labels'].numpy()
        batch_size, _ = log_probs.shape
        dataset_size += batch_size
        # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
        rankings = (-log_probs).argsort(axis=1)
        for ranking, label in zip(rankings, labels):
            if label == ignore_index:
                ignored_predictions += 1
                continue
            if ranking[0] == label:
                hits_at_1 += 1
            # +1 to count from 1 instead of 0
            rank = (ranking == label).nonzero()[0].item() + 1
            mrr += 1/rank    
    # metrics["MRR@N*M"] = mrr / (dataset_size-ignored_predictions)
    # metrics["hits@1"] = hits_at_1 / (dataset_size-ignored_predictions)
    metrics["MRR@NM"] = mrr / (dataset_size-ignored_predictions)
    metrics["hits@1"] = hits_at_1 / (dataset_size-ignored_predictions)

    return metrics

def retrieval_oldest(eval_prediction, ignore_index=-100):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_prediction: EvalPrediction (dict-like)
        predictions: np.ndarray
            shape (dataset_size, N*M)
            This corresponds to the log-probability of the relevant passages per batch (N*M == batch size)
        label_ids: np.ndarray
            shape (dataset_size, )
            Label at the batch-level (each value should be included in [0, N-1] inclusive)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """

    # print(f"eval_prediction.predictions.shape: {eval_prediction.predictions.shape}")
    # print(f"               .label_ids.shape: {eval_prediction.label_ids.shape}")
    metrics = {}  
    
    # log_probs =  torch.stack( [ TENSOR for eval_prediction_element in eval_prediction for TENSOR in eval_prediction_element[1]['log_probs'].detach().cpu() ] ).numpy()
    # label_ids =  torch.stack( [ TENSOR for eval_prediction_element in eval_prediction for TENSOR in eval_prediction_element[1]['label_ids'].detach().cpu() ] ).numpy()

    log_probs =  [ TENSOR for eval_prediction_element in eval_prediction for TENSOR in eval_prediction_element[1]['log_probs'].detach().cpu().numpy() ] 
    label_ids =  [ TENSOR for eval_prediction_element in eval_prediction for TENSOR in eval_prediction_element[1]['label_ids'].detach().cpu().numpy() ]   

    # log_probs = eval_prediction.predictions

    # dataset_size, N_times_M = log_probs.shape
    dataset_size = len(log_probs)
    
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)    

    rankings =  [ (-lp).argsort(axis=-1) for lp in log_probs ] 
    # print(rankings)
    # rankings = (-log_probs).argsort(axis=1)

    mrr, ignored_predictions = 0, 0
    # for ranking, label in zip(rankings, eval_prediction.label_ids):
    for ranking, label in zip(rankings, label_ids):    
        if label == ignore_index:
            ignored_predictions += 1
            continue
        # +1 to count from 1 instead of 0
        rank = (ranking == label).nonzero()[0].item() + 1
        mrr += 1/rank
    mrr /= (dataset_size-ignored_predictions)
    # print(f"dataset_size: {dataset_size}, ignored_predictions: {ignored_predictions}")
    metrics["mrr"] = mrr

    # argmax to get index of prediction (equivalent to `log_probs.argmax(axis=1)`)
    # predictions = rankings[:, 0]
    predictions =  [ r[0] for r in rankings ]

    # print(f"predictions[:100] {predictions.shape}:\n{predictions[:100]}")
    # print(f"eval_prediction.label_ids[:100] {eval_prediction.label_ids.shape}:\n{eval_prediction.label_ids[:100]}")
    # hits@1

    # where = eval_prediction.label_ids != ignore_index
    # metrics["hits@1"] = (predictions[where] == eval_prediction.label_ids[where]).mean()

    where = label_ids != ignore_index
    metrics["hits@1"] = (predictions[where] == label_ids[where]).mean()

    return metrics


def retrieval_(eval_outputs, ignore_index=-100, output_key='log_probs'):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains log_probs and labels for all batches in the evaluation step (either validation or test)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    output_key: str, optional
        Name of the model output in eval_outputs
    """
    metrics = {}    
    mrr, hits_at_1, ignored_predictions, dataset_size = 0, 0, 0, 0
    for batch in eval_outputs:
        log_probs = batch[output_key].numpy()
        labels = batch['labels'].numpy()        
        batch_size, _ = log_probs.shape
        dataset_size += batch_size
        # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
        rankings = (-log_probs).argsort(axis=1)
        for ranking, label in zip(rankings, labels):
            if label == ignore_index:
                ignored_predictions += 1
                continue
            if ranking[0] == label:
                hits_at_1 += 1
            # +1 to count from 1 instead of 0
            rank = (ranking == label).nonzero()[0].item() + 1
            mrr += 1/rank    
    metrics["MRR@NM"] = mrr / (dataset_size-ignored_predictions)
    metrics["hits@1"] = hits_at_1 / (dataset_size-ignored_predictions)

    return metrics


def get_run(eval_outputs, ir_run):
    """    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains logits for all batches in the evaluation step (either validation or test)
    ir_run: ranx.Run
        Original IR run being re-ranked.
    """
    run = {}
    for batch in eval_outputs:
        logits = batch['logits'].numpy()
        N, M = logits.shape
        question_ids = [batch['ids'][i] for i in range(0, N*M, M)]
        rankings = (-logits).argsort(axis=1)
        for ranking, logit, question_id in zip(rankings, logits, question_ids):
            ir_results = ir_run.run[question_id]
            # nothing to re-rank. 
            # this can happen if searching for something unavailable in the query
            # e.g. no face was detected but you are searching for face similarity (see ir.search)
            if not ir_results:
                run[question_id] = ir_results
            else:
                doc_ids = list(ir_results.keys())[: M]
                run[question_id] = {doc_ids[i]: logit[i] for i in ranking}
    return ranx.Run(run)


def f1_score(prediction, ground_truth):
    prediction_tokens = answer_preprocess(prediction).split()
    ground_truth_tokens = answer_preprocess(ground_truth).split()                                                                                                                                                  
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return answer_preprocess(prediction) == answer_preprocess(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def squad(predictions, references):
    """
    Adapted from datasets.load_metric('squad')
    
    Parameters
    ----------
    predictions: List[str]
    references: List[List[str]]
    """

    assert len(predictions) == len(references)
    f1, exact_match = 0, 0
    for prediction, ground_truths in zip(predictions, references):
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / len(references)
    f1 = f1 / len(references)

    return {"exact_match": exact_match, "f1": f1}




def fuse_qrels(qrels_paths):
    """
    Loads all qrels in qrels_paths and unions them under a single Qrels.
    
    Parameters
    ----------
    qrels_paths: List[str]
    
    Returns
    -------
    fused_qrels: ranx.Qrels
    """
    # nothing to fuse
    if len(qrels_paths) == 1:
        return ranx.Qrels.from_file(qrels_paths[0])
    final_qrels = {}
    for i, qrels_path in tqdm(enumerate(qrels_paths)):
        qrels = ranx.Qrels.from_file(qrels_path).qrels
        for q_id, rels in qrels.items():
            final_qrels.setdefault(q_id, {})
            for doc_id, score in rels.items():
                if doc_id in final_qrels[q_id] and final_qrels[q_id][doc_id] != score:
                    raise ValueError(
                        f"{qrels_path} contradicts a prior Qrels (one of {qrels_paths[:i]}).\n"
                        f"Got {score} and {final_qrels[q_id][doc_id]} "
                        f"for question '{q_id}' and document '{doc_id}'"
                    )
                final_qrels[q_id][doc_id] = score
    return ranx.Qrels.from_dict(final_qrels)