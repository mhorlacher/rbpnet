# %%
import sys

# %%
import yaml
import pysam
import pyBigWig

# %%
import numpy as np
import tensorflow as tf

from rbpnet.utils import nan_to_zero, sequence2int, reverse_complement

# %%
class Fasta():
    def __init__(self, fasta) -> None:
        self._fasta = pysam.FastaFile(fasta)
    
    def fetch(self, chrom, start, end, strand='+'):
        sequence = self._fasta.fetch(chrom, start, end)

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        return sequence

    def window(self, chrom, center, strand='+', size=400):
        start, end = center - int(size/2), center + int(size/2) + (size % 2)
        return self.fetch(chrom, start, end, strand)

# %%
def _bytes_features(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# %%
class Site:
    """Class holding cross-link site information."""

    def __init__(self, chrom, start, end, name=None, score=None, strand=None) -> None:
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
        self.name = name

        try: 
            self.score = float(score)
        except ValueError:
            self.score = 0.0
    
    def __repr__(self) -> str:
        return f'{self.chrom}:{self.start}-{self.end}:{self.strand}:{self.name}'

class BigWig:
    def __init__(self, bigWigFile) -> None:
        self._bigWig = bigWigFile
    
    def values(self, chrom, start, end):
        # lazy-load bigWig
        if not isinstance(self._bigWig, pyBigWig.pyBigWig):
            self._bigWig = pyBigWig.open(self._bigWig)
        
        try:
            return self._bigWig.values(chrom, start, end)
        except RuntimeError as e:
            print('WARNING: ' + str(e), file=sys.stderr)
            print(f'WARNING: ({chrom}\t{start}\t{end})', file=sys.stderr)
            return [0.0]*(end-start)

# %%
class Track:
    def __init__(self, bigWigPlus, bigWigMinus) -> None:
        self._bigWigPlus = BigWig(bigWigPlus)
        self._bigWigMinus = BigWig(bigWigMinus)

    def profile(self, chrom, start, end, strand, reverse=True):
        """Return the signal profile for a given sample. 

        Args:
            chrom  (str): Chromosome (chr1, chr2, ...)
            start  (int): 0-based start position
            end    (int): 0-based end position
            strand (str): Strand ('+' or '-')

        Raises:
            ValueError: [description]

        Returns:
            numpy.array: Numpy array of shape (end-start, 1)
        """

        if strand == '+':
            bigWig = self._bigWigPlus
        elif strand == '-':
            bigWig = self._bigWigMinus
        else:
            raise ValueError(f'Unspected strand: {strand}')

        profile = bigWig.values(str(chrom), start, end)
        profile = [nan_to_zero(c) for c in profile]

        if strand == '-' and reverse:
            profile = list(reversed(profile))

        profile = np.array(profile, dtype=np.float32)

        return profile
    
    def window(self, chrom, center, strand='+', reverse=False, size=201):
        start, end = center - int(size/2), center + int(size/2) + (size % 2)
        return self.profile(chrom, start, end, strand, reverse=reverse)

# %%
class TaskSpec:
    """Class holding bigWig signal and cross-link site (peak) information. 
    """

    def __init__(self, task_dict) -> None:
        self.bedfile = task_dict['peaks']
        self.main = self._create_track(task_dict['main'])

        # add control track, if present
        if 'control' in task_dict:
            self.control = self._create_track(task_dict['control'])
        else:
            self.control = None
    
    def _create_track(self, track_dict):
        return Track(track_dict[0], track_dict[1])
    
    @property
    def sites(self):
        """Yields Site instances for the given task.

        Yields:
            Site: A site of the given task. 
        """

        with open(self.bedfile) as f:
            for line in f:
                row = line.strip().split('\t')
                yield Site(*row[:6])
                
# %%
class DataSpec:
    def __init__(self, dataspec_yaml) -> None:
        with open(dataspec_yaml) as f:
            dataspec_dict = yaml.load(f, yaml.CLoader)
        
        self._fasta = dataspec_dict['fasta_file']
        
        self.tasks = {}
        for task in dataspec_dict['task_specs']:
            self.tasks[task] = TaskSpec(dataspec_dict['task_specs'][task])
    
    def sequence(self, chrom, start, end, strand):
        # lazy-load fasta
        if not isinstance(self._fasta, pysam.FastaFile):
            self._fasta = pysam.FastaFile(self._fasta)

        sequence = list(self._fasta.fetch(chrom, start, end))

        if strand == '+':
            pass
        elif strand == '-':
            sequence = list(reverse_complement(sequence))
        else:
            raise ValueError(f'Unspected strand: {strand}')

        return sequence
    
    @property
    def sites(self):
        for task in self.tasks:
            for site in self.tasks[task].sites:
                yield site

# %%
class Sample:
    def __init__(self, site, dataspec, window_size) -> None:
        self.site = site
        self.dataspec = dataspec
        self.window_size = window_size

        # compute extended start/end
        if self.window_size > 1:
            self.extended_start = self.site.start - (int(self.window_size / 2) + (self.window_size % 2)) +1
            self.extended_end = self.site.start + (int(self.window_size / 2)) + 1
            assert (self.extended_end - self.extended_start) == self.window_size
        elif self.window_size == -1:
            # ignore window_size and use BED ranges as provided by user
            self.extended_start = self.site.start
            self.extended_end = self.site.end
        else:
            raise ValueError(f'Unspected window_size: {self.window_size}')

    @property
    def serialized(self):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """

        feature = dict()
        feature['name'] = _bytes_features([self.name.encode('UTF-8')])
        # feature['score'] = _float_feature(np.array(self.score, dtype=np.float32))
        feature['score'] = _float_feature(float(self.score))
        feature['sequence'] = _bytes_features([tf.io.serialize_tensor(self.sequence).numpy()])

        for profile_name, profile in self.profiles.items():
            feature[profile_name] = _bytes_features([tf.io.serialize_tensor(profile).numpy()])

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @property
    def name(self):
        """Return the sample name."""

        return repr(self.site)
    
    @property
    def location(self):
        location_dict = {
            'chrom': self.site.chrom,
            'start': self.extended_start,
            'end': self.extended_end,
            'strand': self.site.strand
        }
        return location_dict
    
    @property
    def score(self):
        """Return the sample score."""

        return self.site.score
    
    @property
    def sequence(self):
        sequence = self.dataspec.sequence(**self.location)
        sequence = sequence2int(sequence)

        # assert that sequnce has the correct length
        assert len(sequence) == (self.extended_end - self.extended_start)

        return sequence
    
    @property
    def profiles(self):
        """Return signal profiles for all tasks.

        Returns:
            dict: Dict of signal profiles (main, control) for each task.
        """

        profiles = {}

        for task in self.dataspec.tasks:
            profiles[task + '_profile'] = self.dataspec.tasks[task].main.profile(**self.location)

            if self.dataspec.tasks[task].control is not None:
                # add control tracks if they are contained in the dataspec
                profiles[task + '_profile_control'] = self.dataspec.tasks[task].control.profile(**self.location)

        return profiles

# %%
class Data:
    def __init__(self, tfrecords, dataspec, use_bias=False) -> None:
        """Holds data(set) information. 

        Args:
            tfrecords ([str]): List of TFRecord filepaths. 
            dataspec (DataSpec): DataSpec instance. 
            use_bias (bool, optional): Indicate whether to use bias tracks. Defaults to False.
        """

        self.tfrecords = tfrecords
        self.dataspec = dataspec
        self.use_bias = use_bias

        self.feature_description = self._make_feature_description()
    
    def _make_feature_description(self):
        """Generate TFRecord feature description(s) based on dataspec. """

        feature_description = {}
        feature_description['sequence'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        feature_description['name'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        feature_description['score'] = tf.io.FixedLenFeature([], tf.float32, default_value=-999.0)

        for task in self.dataspec.tasks:
            feature_description[f'{task}_profile'] = tf.io.FixedLenFeature([], tf.string, default_value='')
            if self.use_bias:
                feature_description[f'{task}_profile_control'] = tf.io.FixedLenFeature([], tf.string, default_value='')

        return feature_description

    @tf.function
    def _parse_sample(self, example_proto):
        """Parse a TFRecord example. """

        # parse example to dict
        parsed_example = tf.io.parse_single_example(example_proto, self.feature_description)

        # parse example info (name, score) and write to info_dict
        info_dict = {'name': parsed_example['name'], 'score': parsed_example['score']}

        #info_dict['name'] = tf.expand_dims(info_dict['name'], 1)

        # parse serialized sequence and write to input_dict
        input_dict = {'sequence': tf.io.parse_tensor(parsed_example['sequence'], tf.int32)}
        input_dict['sequence'] = tf.one_hot(input_dict['sequence'], depth=4)
        tf.debugging.assert_rank(input_dict['sequence'], 2)

        # parse output head data (profiles, counts) and write to output_dict
        output_dict = {}
        for task in self.dataspec.tasks:
            output_dict[f'{task}_profile'] = tf.io.parse_tensor(parsed_example[f'{task}_profile'], tf.float32)
            tf.debugging.assert_rank(output_dict[f'{task}_profile'], 1)

            if self.use_bias:
                output_dict[f'{task}_profile_control'] = tf.io.parse_tensor(parsed_example[f'{task}_profile_control'], tf.float32)
                tf.debugging.assert_rank(output_dict[f'{task}_profile_control'], 1)

        return (info_dict, input_dict, output_dict)

    @tf.function
    def _load_tfrecord_dataset(self, filepath):
        """Load a TFrecord dataset from filepath."""

        return tf.data.TFRecordDataset(filepath)

    def dataset(self, batch_size=128, shuffle=0, cache=True, return_info=False):
        """Create a tf.data.Dataset. 

        Args:
            batch_size (int, optional): Batch size. Defaults to 128.
            shuffle (int, optional): Shuffle buffer. Defaults to 0.
            cache (bool, optional): Indicate whether to cache example_proto's in memory. Defaults to True.
            return_info (bool, optional): Indivate whether to return sample info (name and score). Defaults to False.

        Returns:
            tf.data.Data: Tensorflow Dataset
        """

        # load tfrecord datasets
        dataset = tf.data.Dataset.from_tensor_slices(self.tfrecords)
        dataset = dataset.interleave(self._load_tfrecord_dataset, cycle_length=len(self.tfrecords), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if cache:
            dataset = dataset.cache()
        
        
        if shuffle > 0:
            dataset = dataset.shuffle(shuffle)
        elif shuffle < 0:
            dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset))
        else:
            # no shuffle
            pass

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        dataset = dataset.map(self._parse_sample)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # parse serialized tfrecord examples
        #dataset = dataset.map(lambda batch: tf.vectorized_map(self._parse_sample, batch), num_parallel_calls=tf.data.AUTOTUNE)
        #dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if not return_info:
            # discard info_dict
            dataset = dataset.map(lambda _, inputs, outputs: (inputs, outputs))
        
        return dataset