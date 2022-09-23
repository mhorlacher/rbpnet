# %%
import tensorflow as tf
import pandas as pd

# %%
import matplotlib.pyplot as plt
import logomaker

from rbpnet import utils, variant_impact

# %%
def make_attribution_figure(a, ax):
    df = pd.DataFrame(a, columns=['A', 'C', 'G', 'U'])
    logomaker.Logo(df, shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold', ax=ax)

# %%
def visualize_track(track, sequence=None, title=None):
    fig, axs = plt.subplots(2, 1, figsize=(22, 5), gridspec_kw={'height_ratios': [5, 0.3]})
    axs[0].set_title(title)
    axs[0].plot(track, color='red', label='Pred. Signal', linewidth=2)
    
    make_attribution_figure(utils.sequence2onehot(sequence).numpy(), axs[1])
    
    for ax in axs:
        # remove x-axis margins
        ax.margins(x=0.005)

    # remove plot boarder (except for x-axis)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].get_xaxis().set_visible(False)

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    #axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    
    return fig

# %%
def visualize_track_attribution(track, attribution, sequence=None, title=None):
    nplots = 3 if sequence is not None else 2
    hratio = [5, 2, 0.3] if sequence is not None else [5, 2]
    
    fig, axs = plt.subplots(nplots, 1, figsize=(22, 5), gridspec_kw={'height_ratios': hratio})
    axs[0].set_title(title)
    axs[0].plot(track, color='red', label='Pred. Signal', linewidth=2)
    
    if isinstance(attribution, tf.Tensor):
        attribution = attribution.numpy()
    make_attribution_figure(attribution, axs[1])
    
    if sequence is not None:
        make_attribution_figure(utils.sequence2onehot(sequence).numpy(), axs[2])
    
    for ax in axs:
        # remove x-axis margins
        ax.margins(x=0.005)

    # remove plot boarder (except for x-axis)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].get_xaxis().set_visible(False)
    
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_visible(False)

    if sequence is not None:
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['left'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        #axs[2].get_xaxis().set_visible(False)
        axs[2].get_yaxis().set_visible(False)
    
    return fig

# %%
def visualize_mutation_impact(model, sequence, position, allele_A, allele_B, reverse_complement=False):
    sequence_A = variant_impact.mutate_sequence(sequence, position, alt_base=allele_A)
    sequence_B = variant_impact.mutate_sequence(sequence, position, alt_base=allele_B)
    
    pred_A = model.predict_from_sequence(sequence_A)['QKI_HepG2_profile_target'].numpy()
    pred_B = model.predict_from_sequence(sequence_B)['QKI_HepG2_profile_target'].numpy()

    impact_score = model.variant_impact(sequence, position, allele_A, allele_B, reverse_complement=False)['QKI_HepG2_profile_target']

    attribution_A = model.explain(utils.sequence2onehot(sequence_A))['QKI_HepG2_profile_target'].numpy()
    attribution_B = model.explain(utils.sequence2onehot(sequence_B))['QKI_HepG2_profile_target'].numpy()

    fig, axs = plt.subplots(4, 1, figsize=(22, 7), gridspec_kw={'height_ratios': [5, 0.4, 2, 2]})
    #axs[0].set_title(title.format(impact_score=impact_score))
    axs[0].plot(pred_A, color='blue', label='Signal [Allele A]', linewidth=2)
    axs[0].plot(pred_B, color='red', label='Signal [Allele B]', linewidth=2)
    axs[0].set_yticks([0.0, 0.04, 0.08])

    # sequence
    make_attribution_figure(utils.sequence2onehot(sequence_A).numpy(), axs[1])
    #axs[1].axis('off')

    make_attribution_figure(attribution_A, axs[2])
    make_attribution_figure(attribution_B, axs[3])

    axs[2].axvline(x=position-0.5, color='black', linestyle='--', linewidth=1.4, alpha=0.5)
    axs[2].axvline(x=position+0.5, color='black', linestyle='--', linewidth=1.4, alpha=0.5)
    axs[3].axvline(x=position-0.5, color='black', linestyle='--', linewidth=1.4, alpha=0.5)
    axs[3].axvline(x=position+0.5, color='black', linestyle='--', linewidth=1.4, alpha=0.5)


    for ax in axs:
        # remove x-axis margins
        ax.margins(x=0.005)

    # remove plot boarder (except for x-axis)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].get_xaxis().set_visible(False)

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)
    axs[2].get_xaxis().set_visible(False)

    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].spines['bottom'].set_visible(False)

    axs[0].legend(loc="upper left", ncol=4)
    
    return fig

# %%
def visualize_additiveMix_prediction(seq_onehot, pred_dict, true_dict, task, attribution=None):
    pred_total = tf.math.softmax(tf.squeeze(pred_dict[f'{task}_profile']))
    pred_target = tf.math.softmax(tf.squeeze(pred_dict[f'{task}_profile_target']))
    pred_control = tf.math.softmax(tf.squeeze(pred_dict[f'{task}_profile_control']))
    pred_mix_coeff = tf.math.sigmoid(tf.squeeze(pred_dict[f'{task}_mixing_coefficient'])).numpy()

    true_total = tf.squeeze(true_dict[f'{task}_profile'])
    true_control = tf.squeeze(true_dict[f'{task}_profile_control'])

    true_total_sum = tf.reduce_sum(true_total)
    true_control_sum = tf.reduce_sum(true_control)

    # plotting
    fig, axs = plt.subplots(4, 1, figsize=(22, 12), gridspec_kw={'height_ratios': [8, 3, 0 if attribution is None else 2, 0.6]})

    # total track
    axs[0].set_title(f'task = {task} | mix. coeff. = {pred_mix_coeff:.3f}')
    axs[0].plot(true_total, color='black', label='eCLIP: true counts [total]', linewidth=2)
    axs[0].plot(pred_total * true_total_sum, color='orange', label='eCLIP: pred counts [total]', linewidth=2)
    axs[0].plot(pred_target * true_total_sum * pred_mix_coeff, color='red', linestyle='--', label='eCLIP: pred counts [target]', linewidth=2)
    axs[0].plot(pred_control * true_total_sum * (1 - pred_mix_coeff), color='green', linestyle='--', label='eCLIP: pred counts [control]', linewidth=2)

    # control track
    axs[1].plot(true_control, color='black', label='eCLIP: true counts [control]', linewidth=2)
    axs[1].plot(pred_control * true_control_sum, color='green', label='eCLIP: pred counts [control]', linewidth=2)

    # attribution
    if attribution is not None:
        make_attribution_figure(attribution, axs[2])
    else:
        axs[2].axis('off')

    # sequence
    make_attribution_figure(tf.squeeze(seq_onehot).numpy(), axs[3])
    axs[3].axis('off')

    for ax in axs:
        # remove x-axis margins
        ax.margins(x=0.00)

        # remove plot boarder (except for x-axis)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # set lengends
    axs[0].legend(loc="upper left", ncol=4)
    axs[1].legend(loc="upper left", ncol=2)

    return fig, axs