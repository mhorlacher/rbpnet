# %%
import sys

# %%
import numpy as np
import tensorflow as tf

import gin

# %%
@gin.configurable
class LRAdjustFactor(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience=5):
        super(LRAdjustFactor).__init__()
        self.patience = patience
        self.factor = factor

        print(f'LRAdjustFactor is scheduled with patience of {self.patience} and factor of {self.factor}.', file=sys.stderr)

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0

        # Initialize the best as infinity.
        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        if not 'val_loss' in logs:
            raise ValueError("No validation loss ('val_loss') found in logs.")
        current = logs.get('val_loss')

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                lr = self.model.optimizer.learning_rate.numpy()
                lr_new = lr * self.factor

                print(f'Patience reached. Adjusting lr from {lr:.8f} to {lr_new:.8f} (factor of {self.factor}).', file=sys.stderr)
                self.model.optimizer.learning_rate = lr_new
