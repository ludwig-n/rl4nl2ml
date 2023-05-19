import typing as tp

import jiwer
import numpy as np
import torchtext

import models


def bleu(hypotheses: list[list[str]], references: list[list[list[str]]], weights: list[float]) -> float:
    """
    Computes the BLEU score of a corpus of hypotheses and references.
    Each hypothesis and reference must be a list of strings, each string representing an individual word.

    :param hypotheses: a list of hypotheses
    :param references: a list of lists of references
    :param weights: the n-gram weights
    :return: the BLEU score
    """
    return torchtext.data.metrics.bleu_score(
        candidate_corpus=hypotheses,
        references_corpus=references,
        max_n=len(weights),
        weights=weights
    )


def wer(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    """
    Computes the word error rate (WER) of a corpus of hypotheses and references.
    Each hypothesis and reference must be a list of strings, each string representing an individual word.
    Unlike BLEU, each hypothesis can only have one reference.
    If multiple hypotheses and references are passed, the substitution/insertion/deletion counts
    for all hypothesis-reference pairs are added up and the WER is then computed using these total counts.

    :param hypotheses: a list of hypotheses
    :param references: a list of references
    :return: the word error rate
    """
    return jiwer.wer(
        reference=[' '.join(words) for words in references],
        hypothesis=[' '.join(words) for words in hypotheses]
    )


def compute_metrics(
    model: models.AttentiveModel,
    comp_names: set[str],
    dataset: list[dict[str, tp.Any]],
    temperature: float,
    bleu_weights: list[float]
) -> dict[str, float]:
    """
    Computes metrics for a ``models.AttentiveModel`` on a validation or test dataset.

    :param model: a ``models.AttentiveModel``
    :param comp_names: a set of the competition names to test on
    :param dataset: a dataset to test on
    :param temperature: the temperature to use during inference
    :param bleu_weights: the n-gram weights used to compute BLEU
    :return: a dictionary with metric names as the keys and their values as the values
    """
    reference_answers = []
    model_answers = []
    min_wer_scores = []
    for name in comp_names:
        samples = [x for x in dataset if x['comp_name'] == name]
        target_list = [[str(token) for token in x['target'].tolist()] for x in samples]
        reference_answers.append(target_list)

        model_output = model.generate(samples[0]['h_0'], return_tokens=True, temperature=temperature)
        str_model_output = [str(token) for token in model_output]
        model_answers.append(str_model_output)

        wer_scores = [wer([str_model_output], [target]) for target in target_list]
        min_wer_scores.extend([min(wer_scores)] * len(samples))

    return {
        'bleu': bleu(model_answers, reference_answers, bleu_weights),
        'minwer': np.mean(min_wer_scores)
    }
