# RBPNet: Sequence-to-Signal Learning for CLIP-Seq Data

## Install:

```
pip install git+https://github.com/mhorlacher/rbpnet.git
```

# Usage:

## Training

Training is performed in three steps. First, a `dataspec.yml` file is created which defines tasks and assigns signal bigWig 
and input regions (e.g. called peaks). Using the `dataspec.yml`, a training dataset in form of TFRecords is generated. 
Finally, training is performed with the `rbpnet train` command. 

### 1. Creating a `dataspec.yml`

For an example, check out [examples/QKI.dataspec.yml](https://github.com/mhorlacher/rbpnet/blob/main/examples/QKI.dataspec.yml). 


### 2. Creating a TFRecord file

Use the `rbpnet tfrecord` command to create a `train.tfrecord` file. 

```
Usage: rbpnet tfrecord [OPTIONS] DATASPEC

Options:
  -o, --output TEXT
  -w, --window-size INTEGER
  -s, --shuffle INTEGER
  -o, --output TEXT
  --help                     Show this message and exit.
```

### 3. Training a RBPNet model

Use the TFRecord file and the `rbpnet train` command to train a RBPNet model. 
While the default parameters are optimized to work well on many eCLIP and iCLIP datasets, individual 
model and training parameters can be overwritten via a `config.gin` file. 
See [examples/config.gin](https://github.com/mhorlacher/rbpnet/blob/main/examples/config.gin) for an example. 

After training, a `model.h5` file can be found in the output directory. 

```
Usage: rbpnet train [OPTIONS] [TRAIN_DATA]...

  Train an RBPNet model.

Options:
  --validation-data TEXT
  -c, --config TEXT
  -d, --dataspec TEXT
  -o, --output TEXT
  --help                  Show this message and exit.
```


## Prediction

Takes as input a FASTA file, model and optional target layer. Returns a FASTA file with additional, nucleotide-wise prediction tracks for each sequence and task. 

```
Usage: rbpnet predict [OPTIONS] FASTA

Options:
  -m, --model TEXT
  --layer TEXT      Target layer, i.e. 'profile', 'profile_target' or 'profile_control'.
```

## Feature Importance / Attribution

Takes as input a FASTA file, model and optional target layer. Returns a FASTA file with additional, nucleotide-wise attribution scores for each sequence and task. 

```
Usage: rbpnet explain [OPTIONS] FASTA

Options:
  -m, --model TEXT
  -o, --output TEXT
  --layer TEXT       Target layer, i.e. 'profile', 'profile_target' or 'profile_control'.
```

## Evaluate
Takes as input a TFRecord file, dataspec, model and optional target layer. Prints TSV-formatted SCC scores between predictions and true counts. 

```
Usage: rbpnet evaluate [OPTIONS] TFRECORD

Options:
  -m, --model TEXT
  -d, --dataspec TEXT
  --layer TEXT         Target layer, i.e. 'profile' or 'profile_control'.
```

## Variant Impact Prediction

```
Usage: rbpnet impact [OPTIONS] VARIANTS

Options:
  -f, --fasta TEXT
  -m, --model TEXT
  --assert-a-is-ref      Assert that allele A is the reference allele.
  --layer TEXT           Target layer, i.e. 'profile', 'profile_target' or
                         'profile_control'.
  --window-size INTEGER
```

`rbpnet impact` takes as positional argument a *variant.tsv* file, with one variant specification per line of the form 
`<chrom>	<position>	<strand>	<allele_A>	<allele_B>	<*OPTIONALS>`, e.g.

```
chr8	38262758	+	C	A
chr9	83969990	-	A	C
```

---

## Pre-trained Models

Trained models are available at https://zenodo.org/records/10185223. 
