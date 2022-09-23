# %%
import numpy as np
import tensorflow as tf

from rbpnet import utils

# %%
KLD = tf.keras.losses.KLDivergence()

# %%
def impact_score(profile_A, profile_B):
    return KLD(profile_A, profile_B).numpy()

# %%
def impact_score_on_dict(pred_A, pred_B):
    return {key: impact_score(pred_A[key], pred_B[key]) for key in pred_A.keys() if ('_profile' in key)}

# %%
def mutate_sequence(sequence, position, alt_base):
    return sequence[:position] + alt_base + sequence[position+1:]

# # %%
# def score_sequence_variant_impact(seq_REF, seq_ALT, model):
#     seq_REF_onehot, seq_ALT_onehot = utils.sequence2onehot(seq_REF), utils.sequence2onehot(seq_ALT)
#     pred_REF, pred_ALT = tf.nn.softmax(model(tf.stack([seq_REF_onehot, seq_ALT_onehot])))
#     return impact_score(pred_REF, pred_ALT)

# %%
def variant_impact(model, sequence, position, base_A, base_B, reverse_complement=False):
    allele_A = mutate_sequence(sequence, position, base_A)
    allele_B = mutate_sequence(sequence, position, base_B)
    
    if reverse_complement:
        allele_A, allele_B = utils.reverse_complement(allele_A), utils.reverse_complement(allele_B)
    
    pred_A, pred_B = model.predict_from_sequence(allele_A), model.predict_from_sequence(allele_B)
    
    return impact_score_on_dict(pred_A, pred_B)