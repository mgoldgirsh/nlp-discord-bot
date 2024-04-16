from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from datasets import load_dataset
from discord_model import DiscordModel
from pathlib import Path
import random


def bleu(truth: list, generated: list):
    """
    Calculates bleu score. Compares the ground truth vs generated responses

    Args:
        truth : a list of truth sentences
        generated : a list of generated sentences

    Returns:
        float: bleu score representing similarity between truth and generated
    """
    ref_bleu = []
    gen_bleu = []
    for sentence in truth:
        gen_bleu.append(sentence.split())
    for sentence in generated:
        ref_bleu.append(sentence.split())
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=cc.method4, auto_reweigh=True)
    return score_bleu


# load the test sentences
test_sentences = random.sample(load_dataset("daily_dialog")['test']['dialog'], 10)

to_generate_from = [test[0] for test in test_sentences]
correct_responses = [test[1] for test in test_sentences]

# load the model
model = DiscordModel(f"{Path(__file__).absolute().parent.parent}")

# print('here', to_generate_from, correct_responses)
# # compute generated responses from test sentences
generated_responses = [model.generate(to_generate, with_history=False) for to_generate in to_generate_from]

# print('herev2', generated_responses, correct_responses)
# # print the bleu score evaluation metric
print("The bleu score is:", bleu(correct_responses, generated_responses))
