## Constructing TFRecords for Model Training

To create tfrecord files for training RBPNet models, three steps need to be performed. 

### 1. Specifying training intervals
First, a set of training intervals need to be specified in BED format. In this example, we use the `enriched-windows.py` script to tile a set of transcripts into windows and subsequently select windows which are enriched in eCLIP truncation events ("counts"). 

#### 1.1 Generate enriched 50nt windows
```
python ../../scripts/enriched-windows.py meta/transcripts.chr19.bed --bigWigPlus signal/PTBP1_HepG2/eCLIP/counts.pos.bw --bigWigMinus signal/PTBP1_HepG2/eCLIP/counts.neg.bw -o PTBP1_HepG2.enriched-windows.bed
```
#### 1.2 Extend window flanks by 125nt in each direction

```
bedtools slop -i PTBP1_HepG2.enriched-windows.bed -g meta/grch38.chr19.genomefile -b 125 > PTBP1_HepG2.enriched-windows.300nt.bed
```

The resuling BED file contains overlapping intervals with the enriched region centered in the middle. 

### 2. Specifying `dataspec.yml`

We next populate the `dataspec.yml` with paths to the eCLIP bigWig files, genome FASTA and our training interval BED file. 

```
fasta_file: meta/grch38.chr19.fasta

task_specs:  

  PTBP1_HepG2:
    main:
      - signal/PTBP1_HepG2/eCLIP/counts.pos.bw # plus-strand bigWig
      - signal/PTBP1_HepG2/eCLIP/counts.neg.bw # minus-strand bigWig
    control:
      - signal/PTBP1_HepG2/control/counts.pos.bw
      - signal/PTBP1_HepG2/control/counts.neg.bw
    peaks: PTBP1_HepG2.enriched-windows.300nt.bed # 300nt intervals
```

### 3. Create tfrecords with `rbpnet tfrecord`

```
rbpnet tfrecord PTBP1_HepG2.dataspec.yml -o PTBP1_HepG2.data.tfrecord
```