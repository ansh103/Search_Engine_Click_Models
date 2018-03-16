from __future__ import division, print_function

try:
    from pyspark import SparkContext, SparkConf
except ImportError:
    import findspark
    findspark.init()
    from pyspark import SparkContext, SparkConf

import bisect
import copy
from collections import defaultdict, namedtuple
from contextlib import contextmanager
import hashlib
import itertools
import json
import keras
import math
from mjolnir.utils import hdfs_exists, hdfs_mkdir, hdfs_open_read, as_local_paths, as_output_file
import numpy as np
import os
import psutil
import pylru
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F, HiveContext, types as T
import random
from scipy import sparse
import shutil
from sklearn.metrics import confusion_matrix
import sklearn.utils
import sys
import time


np.random.seed(0)
R = random.Random(0)

SERP_SIZE = 10
# 1 << SERP_SIZE possible click patterns, and we have to record for each possible position
# SERP_SIZE can't be too big. 10 gives 10240, but 20 gives 21M.
CLICK_PATTERN_SIZE_1D = 1 << SERP_SIZE
CLICK_PATTERN_SIZE_2D = SERP_SIZE * CLICK_PATTERN_SIZE_1D

Q_VEC_SIZE = CLICK_PATTERN_SIZE_1D
Q_VEC_START = 0
Q_VEC_END = Q_VEC_START + Q_VEC_SIZE

I_VEC_SIZE = 1
I_VEC_START = Q_VEC_END
I_VEC_END = I_VEC_START + I_VEC_SIZE

D_VEC_SIZE = CLICK_PATTERN_SIZE_2D
D_VEC_START = I_VEC_END
D_VEC_END = D_VEC_START + D_VEC_SIZE

VEC_SIZE = D_VEC_END


def get_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def elapsed_gen(title, report_every=1):
    took = []

    @contextmanager
    def f():
        start = time.time()
        yield
        if report_every <= 1:
            print(title, (time.time() - start) * 1000, 'ms')
        else:
            took.append(time.time() - start)
            if len(took) >= report_every:
                print(title, 'mean', np.mean(took) * 1000, 'ms', 'sum', np.sum(took) * 1000, 'ms')
                del took[:]

    return f


def elapsed_deco(title, report_every=1):
    def inner(f):
        elapsed = elapsed_gen(title, report_every)
        def fn(*args, **kwargs):
            with elapsed():
                return f(*args, **kwargs)
        return fn
    return inner


def sort_hits(hits):
    ordered = sorted(hits, key=lambda hit: hit.hit_position)
    for i, hit in enumerate(ordered[:SERP_SIZE]):
        yield (i, hit.hit_page_id, hit.clicked)


def index_of_click_pattern(clicks):
    return sum([1 << i for i, clicked in enumerate(clicks[:SERP_SIZE]) if clicked])


class SparseVectorBuilder(object):
    def __init__self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype
        # TODO supprt n-dimensions
        self.stride = (shape[-1], 1)
        self.size = shape[0] * shape[1]
        self.indices = []
        self.data = []

    def __setitem(self, key, value):
        assert len(key) == len(self.stride)
        # TODO: support n-dimensions
        if isinstance(key[1], (np.ndarray, list, tuple)):
            assert isinstance(item, (np.ndarray, list, tuple))
            for idx, val in zip(key[2], item):
                key_1d = sum(a*b for a, b in zip(stride, (key[0], idx))
                self.indices.append(key_1d)
                self.data.append(value)
        else:
            key_1d = sum(a*b for a, b in zip(stride, key))
            self.indices.append(key_1d)
            self.data.append(value)

    def build(self):
        vec = Vectors.sparse(self.size, (self.indices, self.data))
        self.clear()
        return vec

    def clear(self):
        del self.indices[:]
        del self.data[:]


class Csr3dBuilder(object):
    """Builder for 3 dimenional sparse matrix in CSR format

    Sadly we convert back and forth several times. Initial data here is
    built in COO format. We convert that to CSR and return it. CSR is
    necessary for slicing, coo_matrix doesn't support extracting batches
    from the full matrix (althogh seems like it could). Providing dense
    arrays to keras means converting back to coo to populate a dense array
    """
    def __init__(self, y_shape, dtype=None):
        assert len(y_shape) == 2
        self.y_shape = y_shape
        self.dtype = dtype
        self.clear()
    def clear(self):
        # TODO: Use numpy arrays and resize? Would cut memory usage vs
        # python list by 1/2.  It might also allow faster copy operations
        # from the vectors
        self.rows = []
        self.cols = []
        self.data = []

    def build(self, n_rows):
        data = (self.data, (self.rows, self.cols))
        shape = (n_rows, self.y_shape[0] * self.y_shape[1])
        return sparse.csr_matrix(data, shape=shape, dtype=self.dtype)

    def __setitem__(self, key, item):
        assert isinstance(key, tuple) and len(key) == 3
        assert isinstance(key[0], int)
        assert isinstance(key[1], int)
        row = key[0]
        offset = self.y_shape[1] * key[1]
        if isinstance(key[2], (np.ndarray, list, tuple)):
            assert isinstance(item, (np.ndarray, list, tuple))
            for idx, val in zip(key[2], item):
                self.rows.append(row)
                self.cols.append(offset + idx)
                self.data.append(val)
        else:
            assert not isinstance(item, (np.ndarray, list, tuple))
            self.rows.append(row)
            self.cols.append(offset + key[2])
            self.data.append(item)


def make_click_pattern_vector(features, size):
    vec = Vectors.sparse(size, features)
    assert min(vec.indices) >= 0
    assert max(vec.indices) < size
    assert min(vec.values) >= 0
    return vec


def build_feature_dataframes(df_deduped, d_group_cols):
    click_pattern_udf = F.udf(index_of_click_pattern, T.IntegerType())
    make_vector_udf = F.udf(make_click_pattern_vector, VectorUDT())

    df_stats = (
        df_deduped
        .withColumn('click_pattern', click_pattern_udf('hits.clicked'))
        .select('wikiid', 'norm_query_id', 'click_pattern', F.explode('hits').alias('hit'))
        .select('wikiid', 'norm_query_id', 'click_pattern', 'hit.hit_page_id', 'hit.hit_position')
        .groupBy('wikiid', 'norm_query_id', 'hit_page_id', 'hit_position', 'click_pattern')
        .agg(F.count(F.lit(1)).alias('count'))
        .cache())

    df_q = (df_stats
        .groupBy('wikiid', 'norm_query_id', 'click_pattern')
        .agg(F.sum('count').alias('count'))
        .groupBy('wikiid', 'norm_query_id')
        .agg(F.struct(F.collect_list('click_pattern').alias('indices'), F.collect_list('count').alias('values')).alias('q_feature'))
        #.withColumn('q_feature', make_vector_udf('q_feature', F.lit(CLICK_PATTERN_SIZE_1D)))
        )

    df_d = (df_stats
        .withColumn('index', (F.col('hit_position') * CLICK_PATTERN_SIZE_1D) + F.col('click_pattern'))
        .groupBy(*(d_group_cols + ['index']))
        .agg(F.sum('count').alias('count'))
        .groupBy(*d_group_cols)
        .agg(F.struct(F.collect_list('index').alias('indices'), F.collect_list('count').alias('values')).alias('d_feature'))
        #.withColumn('d_feature', make_vector_udf('d_feature', F.lit(CLICK_PATTERN_SIZE_2D)))
        )

    return df_q, df_d


def hash_csr_matrix(mat):
    m = hashlib.md5()
    m2 = hashlib.md5()
    for i, d in sorted(zip(mat.indices, mat.data)):
        m.update(str(i).encode('utf8'))
        m2.update(str(i).encode('utf8'))
        m.update(str(d).encode('utf8'))
        m2.update(str(d).encode('utf8'))
    basic = m.hexdigest()
    # indptr varies depending on order of creation. We only care
    # that all the same rows exist, their order is unimportant.
    # Change into sequence size so its comparable across orders
    prev = None
    lengths = []
    for x in sorted(mat.indptr):
        m2.update(str(x).encode('utf8'))
        if prev is not None:
            lengths.append(x - prev)
        prev = x
    for l in sorted(lengths):
        m.update(str(l).encode('utf8'))
    return basic, m.hexdigest(), m2.hexdigest()


def build_sparse_vectors_from_rows(rows):
    x = SparseVectorBuilder((1 + SERP_SIZE, VEC_SIZE), dtype=np.float32)
    y = SparseVectorBuilder((1 + SERP_SIZE, 1), dtype=np.float32)
    for row in enumerate(rows):
        x[0, Q_VEC_START + np.asanyarray(row.q_feature.indices)] = row.q_feature.values
        clicked_prev = False
        for j, hit in enumerate(ordered, 1):
            x[i, j, D_VEC_START + np.asanyarray(hit.d_feature.indices)] = hit.d_feature.values
            if clicked_prev:
                x[i, j, I_VEC_START] = 1
            if hit.clicked:
                y[i, j, 0] = 1
            clicked_prev = hit.clicked
        # One downside is these cary no shape information, it has to
        # be assumed when used.
        yield x.build(), y.build()


def build_matrix_from_rows(rows, builder='coo'):
    x = Csr3dBuilder((1 + SERP_SIZE, VEC_SIZE), dtype=np.float32)
    y = Csr3dBuilder((1 + SERP_SIZE, 1), dtype=np.float32)

    for i, row in enumerate(rows):
        x[i, 0, Q_VEC_START + np.asanyarray(row.q_feature.indices)] = row.q_feature.values
        clicked_prev = False
        # Sadly we have no guarantee these are still in order, although
        # they often are.
        ordered = sorted(row.hits, key=lambda hit: hit.hit_position)
        for j, hit in enumerate(ordered, 1):
            x[i, j, D_VEC_START + np.asanyarray(hit.d_feature.indices)] = hit.d_feature.values
            if clicked_prev:
                x[i, j, I_VEC_START] = 1
            if hit.clicked:
                y[i, j, 0] = 1
            clicked_prev = hit.clicked

    yield x.build(i + 1), y.build(i + 1)


def matrix_to_hdfs(output_dir, prediction_type):
    def fn(partition_id, rows):
        for i, matrices in enumerate(rows):
            names = ('x', 'y')
            output_names = ['part-%05d-%03d-%s-%s.npz' % (partition_id, i, prediction_type, name) for name in names]
            output_paths = [os.path.join(output_dir, x) for x in output_names]
            # TODO: Output a single file with all using np.savez
            # But then we have to dig (only a little) into the sparse matrix implementation
            for path, matrix in zip(output_paths, matrices):
                with as_output_file(path) as f:
                    sparse.save_npz(f, matrix, compressed=True)
            yield {
                'paths': output_paths,
                'shapes': [mat.shape for mat in matrices],
            }

    return fn


@elapsed_deco('Build Dataset')
def build_spark_dataset(df, output_dir, prediction_type):
    hits_schema = T.ArrayType(T.StructType([
        T.StructField('hit_position', T.IntegerType()),
        T.StructField('hit_page_id', T.IntegerType()),
        T.StructField('clicked', T.BooleanType())
    ]))
    sort_hits_udf = F.udf(sort_hits, hits_schema)

    # deduplicate serps seen by the user multiple times, and batch up into
    # a group of hits per session
    df_deduped = (
        df
        .groupBy('wikiid', 'norm_query_id', 'session_id', 'hit_page_id')
        .agg(F.mean('hit_position').alias('hit_position'), F.max('clicked').alias('clicked'))
        .groupBy('wikiid', 'norm_query_id', 'session_id')
        .agg(F.collect_list(F.struct('hit_position', 'hit_page_id', 'clicked')).alias('hits'))
        #.sample(False, 0.1, seed=0)
        .withColumn('hits', sort_hits_udf('hits'))
        .cache())

    print('Observations:', df_deduped.select(F.sum(F.size('hits')).alias('sum')).collect()[0].sum)

    # TODO: Verify conclusions from research paper match our data.
    # Although we don't care that much about click prediction. I suppose
    # though we could try and compare results from a real AB test with
    # that of a test that uses predicted clicks. See if they match up?
    d_group_cols = {
        # Click prediction works best with document vectors aggregated
        # across all queries.
        'click': ['wikiid', 'norm_query_id'],
        # Relevance prediction works best with document vectors
        # aggregated across a single query.
        'relevance': ['wikiid', 'norm_query_id', 'hit_page_id'],
    }[prediction_type]

    df_q, df_d = build_feature_dataframes(df_deduped, d_group_cols)

    df_grouped = (
        df_deduped
        .select('wikiid', 'session_id', 'norm_query_id', F.explode('hits').alias('hit'))
        .select('wikiid', 'session_id', 'norm_query_id', 'hit.*')
        .join(df_d, on=d_group_cols)
        .groupBy('wikiid', 'norm_query_id', 'session_id')
        .agg(F.collect_list(F.struct('hit_position', 'clicked', 'd_feature')).alias('hits'))
        .join(df_q, on=['wikiid', 'norm_query_id']))

    print('Start collecting training matrix')
    start_time = time.time()

    if output_dir:
        res = (
            df_grouped
            # Somehow we need to prevent data leakage between test and train, which basically
            # means either splitting the data early, or partitioning here so that there is
            # no leakage between partitions.
            # This means our sessions are not randomly distributed throughout training though
            .repartition('wikiid', 'norm_query_id')
            .rdd.mapPartitions(build_matrix_from_rows)
            .mapPartitionsWithIndex(matrix_to_hdfs(output_dir, prediction_type))
            .collect())
        with as_output_file(os.path.join(output_dir, 'stats.json'), mode='w') as f:
            f.write(json.dumps(res))
        return res
    else:
        x, y = zip(*df_grouped.rdd.mapPartitions(build_matrix_from_rows).collect())
        x, y = [sparse.vstack(z) for z in (x, y)]

        print('x', hash_csr_matrix(x))
        print('y', hash_csr_matrix(y))

        return sklearn.utils.shuffle(x, y, random_state=np.random.RandomState(0))


def build_training_data(input_dir, output_dir, prediction_type):
    if output_dir:
        if hdfs_exists(output_dir):
            return output_dir
        hdfs_mkdir(output_dir)

    print('Loading training data')
    start_data = time.time()
    try:
        conf = SparkConf()
        conf.set('spark.sql.shuffle.partitions', 2)
        with SparkContext.getOrCreate(conf=conf) as sc:
            sqlContext = HiveContext(sc)
            df = sqlContext.read.parquet(input_dir)
            return build_spark_dataset(df, output_dir, prediction_type)
    except:
        # nuke output dir on failure, so re-runs don't complain
        if output_dir:
            shutil.rmtree(output_dir)
        raise
    finally:
        print('Elapsed: ', time.time() - start_data)


def test_train_split(x, frac):
    train_size = int(x.shape[0] * frac)
    test_size = x.shape[0] - train_size
    return x[0:train_size,:], x[train_size:,:]


class FromHdfs(keras.utils.Sequence):
    def __init__(self, partitions, cache_size=2, shuffle=True):
        self.partitions = partitions
        self.indices = np.arange(len(partitions))
        self.cache = pylru.lrucache(cache_size)
        self.shuffle = shuffle
        self.loads = 0
        self.children = []
        self.loaded_paths = {}

    def __len__(self):
        return len(self.partitions)

    def _load_partition(self, partition_id):
        paths = self.partitions[partition_id]['paths']
        with as_local_paths(paths) as local_paths:
            yield local_paths

    def __getitem__(self, key):
        key = self.indices[key]
        if key not in self.cache:
            # TODO: Instead of these generators and excessive error handling, what if
            # we copy the whole directory at the outset and use a contextmanager to
            # clean it up?
            if key not in self.loaded_paths:
                self.loads += 1
                if self.loads > 1 + len(self.partitions):
                    print('WARNING: Loading partitions multiple times per epoch')
                    # Mute warning until next epoch
                    self.loads = 0 - float('inf')
                child = self._load_partition(key)
                self.children.append(child)
                self.loaded_paths[key] = child.send(None)
            local_paths = self.loaded_paths[key]
            self.cache[key] = tuple(sparse.load_npz(path) for path in local_paths)
        return self.cache[key]

    def on_epoch_end(self):
        self.loads = 0
        # we can't allow keras to shuffle the entire set of minibatches each epoch,
        # so we try and help a little by shuffling the partitions
        if self.shuffle:
            np.random.shuffle(self.indices)

    def clear(self):
        self.loaded_paths.clear()
        self.cache.clear()
        errors = []
        for child in self.children:
            try:
                child.send(None)
            except StopIteration:
                pass
        del self.children[:]
        if errors:
            raise errors[0]


class Reshaper(keras.utils.Sequence):
    """Reshape sparse 2d matrix into dense 3d"""
    def __init__(self, nested, batch_size, x_shape=(-1, 1 + SERP_SIZE, VEC_SIZE), y_shape=(-1, 1 + SERP_SIZE, 1)):
        self.nested = nested
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_batch = np.zeros((batch_size, x_shape[1] * x_shape[2]), dtype=np.float32)
        self.y_batch = np.zeros((batch_size, y_shape[1] * y_shape[2]), dtype=np.float32)
        self.sample_weight = np.ones((batch_size, y_shape[1]), dtype=np.float32)
        self.sample_weight[:,0] = 0


    def __len__(self):
        return len(self.nested)

    # @elapsed_deco('Fetch and Reshape', 883)
    def __getitem__(self, key):
        generator_output = self.nested[key]
        if len(generator_output) == 2:
            x, y = generator_output
            # Not sure this really goes here, but this sets the weight of the first time step of
            # each observation to 0. That time step contains the query and no document, so doesn't
            # predict clicks.
            if y is None:
                sample_weight = None
            else:
                sample_weight = self.sample_weight[:y.shape[0]]
        elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
        else:
            raise ValueError

        batch_size = x.shape[0]
        assert batch_size <= self.batch_size
        x_batch = self.x_batch[:batch_size]
        y_batch = self.y_batch[:batch_size]

        # out=x_batch reuses an array over and over. For some reason this is
        # ~20% slower than creating new arrays on every iteration in my tests,
        # but it also cuts memory usage by 2/3. More curious is time spent in
        # *this* function decreases, but ms/step in keras increases.
        x_batch = x.toarray(out=x_batch).reshape(self.x_shape)
        if y is not None:
            y_batch = y.toarray(out=y_batch).reshape(self.y_shape)
        return x_batch, y_batch, sample_weight

    def on_epoch_end(self):
        self.nested.on_epoch_end()


class HdfsMiniBatch(keras.utils.Sequence):
    def __init__(self, nested, batch_size, shuffle=False):
        self.nested = nested
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.empty(len(self), dtype=np.int32)
        self.update_partition_offsets()

    def __len__(self):
        return int(sum(math.ceil(x['shapes'][0][0] / self.batch_size) for x in self.nested.partitions))

    def __getitem__(self, key):
        # left most value < key. This is necessary because the partitions
        # do not have a uniform length to calculate partition from.
        partition_key = bisect.bisect_left(self.partitions, key) - 1
        generator_output = self.nested[partition_key]
        if len(generator_output) == 3:
            x, y, sample_weights = generator_output
        elif len(generator_output) == 2:
            x, y = generator_output
            sample_weights = None
        else:
            raise ValueError()

        start = self.indices[key]
        end = start + self.batch_size
        x_batch = x[start:end]
        y_batch = y[start:end]
        sw_batch = None if sample_weights is None else sample_weights[start:end]
        return x_batch, y_batch, sw_batch

    def on_epoch_end(self):
        self.nested.on_epoch_end()
        if self.shuffle or self.nested.shuffle:
            self.update_partition_offsets()

    def update_partition_offsets(self):
        self.partitions = []
        start = 0
        for idx in self.nested.indices:
            partition = self.nested.partitions[idx]
            partition_len = partition['shapes'][0][0]
            partition_indices = np.arange(0, partition_len, self.batch_size)
            end = start + partition_indices.shape[0]
            if self.shuffle:
                np.random.shuffle(partition_indices)
            self.indices[start:end] = partition_indices
            self.partitions.append(start)
            start += partition_indices.shape[0]


class MiniBatch(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, key):
        start = key * self.batch_size
        end = start + self.batch_size
        x_batch = self.x[start:end]
        y_batch = self.y[start:end]
        return x_batch, y_batch


@contextmanager
def input_generators(data, frac, batch_size):
    clear = []
    if isinstance(data[0], (sparse.spmatrix, np.ndarray)):
        shuffle = True
        x, y = data
        x_train, x_test = test_train_split(x, frac)
        y_train, y_test = test_train_split(y, frac)
        generator_train = [(x_train, y_train)]
        generator_train = MiniBatch(x_train, y_train, batch_size)
        generator_test = [(x_test, y_test)]
    else:
        # shuffling minibatches would mean grabbing batches from many
        # different files. for reasonable efficiency we need to read
        # one file completely before going on to the next. Instead
        # we have FromHdfs and HdfsMiniBatch each shuffle, which results
        # in a different order of partitions and different order within
        # partitions. Not as good as global shuffle but "good enough".
        shuffle = False
        if isinstance(data, str):
            with hdfs_open_read(os.path.join(data, 'stats.json')) as f:
                partitions = json.load(f)
        else:
            partitions = data

        # Naive test/train split. This can be wildly off for small numbers
        # of partitions. Also the partitions arn't exactly equal size(but close).
        # Worst case of two partitions you always get half on each side.
        partitions = sorted(partitions, key=lambda _: R.random())
        split_size = int(frac * len(partitions))
        data_train = partitions[:split_size]
        data_test = partitions[split_size:]
        print("Using %d (of %d) partitions for training" % (data_train.shape[0], len(partitions)))
        print("Using %D (of %d) partitions for testing" % (data_test.shape[0], len(partitions)))

        generator_train = FromHdfs(data_train)
        clear.append(generator_train)
        generator_train = HdfsMiniBatch(generator_train, batch_size)
        generator_test = FromHdfs(data_test, len(data_test), shuffle=False)
        clear.append(generator_test)
        generator_test = HdfsMiniBatch(generator_test, batch_size, shuffle=False)

    try:
        yield (
            # Reshape sparse 2d batches into dense 3d batches
            Reshaper(generator_train, batch_size),
            Reshaper(generator_test, batch_size),
            shuffle)
    finally:
        for gen in clear:
            gen.clear()


def build_model(
    learning_rate=0.1, activation='tanh', units=32,
    n_hidden=0, reduce_input_dim=True, reduce_activation=None,
):
    from keras import layers
    from keras.models import Sequential
    from keras.optimizers import Adadelta

    model = Sequential()
    input_shape={'input_shape': (1 + SERP_SIZE, VEC_SIZE)}
    if reduce_input_dim:
        # Reduce our input dimensionality. Not ideal but saves a ton of training time.
        # Probably makes accuracy worse? But seems about the same on a toy dataset.
        # TODO: Evaluate with larger datasets
        model.add(layers.TimeDistributed(layers.Dense(
            units=reduce_input_dim, activation=None), name='reduce_dim', **input_shape))
        if reduce_activation:
            model.add(layers.Activation(reduce_activation, name=reduce_activation))
        input_shape.clear()

    # LSTM treats sequence of hits on SERP of one observation as a sequence of time steps
    for _ in range(n_hidden+1):
        model.add(layers.LSTM(units, return_sequences=True, activation=None, **input_shape))
        input_shape.clear()
        if activation:
            model.add(layers.Activation(activation, name=activation))
    # For each hit predict a single output, was the result clicked?
    # TODO: mask out the first element of time series, it is only the query and has nothing to predict
    model.add(layers.TimeDistributed(layers.Dense(units=1, activation=None), name='output'))
    model.add(layers.Activation('sigmoid', name='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adadelta(lr=learning_rate, clipnorm=1.0),
        sample_weight_mode='temporal',
        metrics=['accuracy'])
    model.summary()

    return model


def confusion(model, generator_test):
    confusion = np.zeros((2, 2), dtype=np.int)
    for i in range(len(generator_test)):
        generator_output = generator_test[i]
        if len(generator_output) == 2:
           x_test_batch, y_test_batch = generator_output
        elif len(generator_output) == 3:
           x_test_batch, y_test_batch, _ = generator_output
        else:
            raise ValueError()
        y_pred = model.predict(x_test_batch)
        # Remove the 0 time step which has no information.
        y_test_batch = y_test_batch[:,1:,:].ravel()
        y_pred = y_pred[:,1:,:].ravel()

        confusion += confusion_matrix(y_test_batch, y_pred.round())
    return confusion


def train(data, frac=0.7, batch_size=None, epochs=150, **kwargs):

    # Can't extract before grid search, batch_size is parameterized
    with input_generators(data, frac, batch_size) as (generator_train, generator_test, shuffle):
        model = build_model(**kwargs)
        history = model.fit_generator(
            generator=generator_train,
            validation_data=generator_test,
            epochs=epochs,
            callbacks=[keras.callbacks.EarlyStopping(patience=10)],
            shuffle=shuffle,
            # TODO: On my laptop (with only 2cpu) this is much slower, worth
            # trying on a high core count machine though.
            # workers=1, use_multiprocessing=True,
            verbose=1)
        print('Optimization Finished!')
        print(confusion(model, generator_test))
        return model, history


def grid_search(data, space):
    product = itertools.product(*space.values())
    output = []
    for kwargs in (dict(zip(space.keys(), x)) for x in product):
        print('Training with args:', kwargs)
        output.append((kwargs,) + train(data, **kwargs))
    return output


def limit_keras_cpus(n_cpus=3):
    from keras import backend as K
    import tensorflow as tf

    config = tf.ConfigProto(
        intra_op_parallelism_threads=n_cpus,
        inter_op_parallelism_threads=n_cpus,
        allow_soft_placement=True,
        device_count = {'CPU': n_cpus})
    session = tf.Session(config=config)
    K.set_session(session)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_dir=sys.argv[1]
        output_dir=sys.argv[2]
    else:
        input_dir='/vagrant/tmp/dbn_input'
        output_dir='/vagrant/fake_data/ncm'

    limit_keras_cpus()
    data = build_training_data(
        input_dir=input_dir, output_dir=output_dir,
        prediction_type='relevance')
    output = grid_search(data, {
        # Early stopping means this needs to be sufficiently large
        'epochs': [150],
        # mini-batches? Or memory sized batches?
        'batch_size': [128, 64],
        # Doesn't seem to help, maybe more training data?
        'n_hidden': [0],
        'learning_rate': [0.1],
        # TODO: try a few
        'activation': [None, 'tanh'],
        # No clue what is appropriate...
        'units': [24],
        # Make a smaller network by dramaticall reducing the input dimension before
        # passing into LSTM (which has 4x as many nodes per input). This allows
        # keeping LSTM wider then if it were a direct connection.
        # For example, (11,11265) input with:
        #   units: 24, reduce_input_dim: 8 gives 93k params
        #   units: 24, reduce_input_dim: False gives 1M params
        #   units: 3, reduce_input_dim: False gives 135k params
        'reduce_input_dim': [8],
    })
    for kwargs, model, history in output:
        print(history)

