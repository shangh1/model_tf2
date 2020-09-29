from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np

feature_delimiter = '\t'

import tensorflow_text
import tensorflow as tf
import collections

from toy_tf_util import MecKaminoFeature

def main_fun(mode):
    """Example demonstrating loading TFRecords directly from disk (e.g. HDFS) without tensorflow_datasets."""
    import tensorflow as tf
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    BUFFER_SIZE = 10000
    BATCH_SIZE = 5
    NUM_WORKERS = 2
    num_classes = 19
    model_dir = './'
    epochs = 10
    mode = mode
    input_file = 'toy_sample.txt'
    val_path_file = 'toy_sample.txt'
    test_path_file = 'toy_sample.txt'

    def line_to_multiple_features(inputString, vocab_file_path, mode, seq_length, num_class):
        inputString = [inputString]
        batch_ids, labels, input_ids = MecKaminoFeature(inputString, vocab_file_path, seq_length, num_class)
        d = {'ids':batch_ids[0], 'input_feature':input_ids[0]}
        if mode == 'train':
            return d, labels[0]
        else:
            return d

    def get_unbatched_feature_label(filename, mode, epochs):
        dataset = tf.data.TextLineDataset(filename)
        if mode == 'train':
            dataset = dataset.repeat(epochs).shuffle(BUFFER_SIZE)
        # convert every line to features
        print('before: unbatch_parse_data:')
        unbatch_parse_data = dataset.map(lambda x: line_to_multiple_features(x, 'vocab.txt', mode, 120,
                                                                             num_classes))
        print('after: unbatch_parse_data:')
        return unbatch_parse_data

    def build_tf_api():
        vocab_size = 30522
        embedding_dim = 300
        num_filters = 512
        filter_sizes = [2,3,4]
        drop = 0.5

        inputs = {'ids':Input(shape=(4, ), dtype='string'),
                  'input_feature':Input(shape=(120, ), dtype='int32')}
        input_feature = inputs['input_feature']
        sequence_length = 120
        print('---------------------------------------')
        print('input_feature:')
        print(input_feature)

        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(input_feature)
        reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)
        
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        logits = Dense(units=num_classes)(dropout)
        pred_probs = tf.keras.activations.sigmoid(logits)
        model = Model(inputs=inputs, outputs=pred_probs)

        def get_loss_fn():
            def get_loss(labels, outputs):
                logits = tf.math.log((outputs+0.0000001) / ( 1 - outputs+0.0000001))
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                # loss has the same shape as logits: 1 loss per class and per sample in the batch
                cost = tf.reduce_mean(
                    tf.reduce_sum(loss, axis=1)
                )
                return cost
            return get_loss

        adam = Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss=get_loss_fn(),
                      metrics=['accuracy'])
        print('Training Model...')
        return model

    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

    # Dataset for input data
    def get_feature_label(_input):
        input_files = tf.data.Dataset.list_files(_input)
        datasets_unbatched = get_unbatched_feature_label(input_files, mode, epochs)
        data_feature_label = datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
        print('data_feature_label:' + str(type(data_feature_label)))
        return data_feature_label

    tf.io.gfile.makedirs(model_dir)
    filepath = model_dir # + "/weights-{epoch:04d}"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode='auto', save_freq=10)]

    steps_per_epoch = 60000 / GLOBAL_BATCH_SIZE * 0.9

    if mode == 'train':
        with strategy.scope():
            multi_worker_model = build_tf_api()
        train_feature_label = get_feature_label(input_file)
        validation_feature_label = get_feature_label(val_path_file)
        histCallback = multi_worker_model.fit(x=train_feature_label, validation_data=validation_feature_label, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        for key in histCallback.history:
            print(key)
        results = multi_worker_model.evaluate(validation_feature_label, verbose=2)
        for name, value in zip(multi_worker_model.metrics_names, results):
            print("%s: %.3f" % (name, value))
        multi_worker_model.save(model_dir)
    else: # if mode is inference.
        def concat_id(a):
            return a[0] + b'|' + a[1] + b'|' + a[2]
        print('now is inference mode')
        test_feature = get_feature_label(test_path_file)
        reconstructed_model = tf.keras.models.load_model(model_dir, compile=False)
        all_id_pred = np.array([[]] * 20).transpose()
        for elem in test_feature:
            ids_np = np.array(elem['ids'])[:,:3]
            ids_np_1dim = np.apply_along_axis(concat_id, 1, ids_np)
            ids_np_1dim = np.reshape(ids_np_1dim, (-1, 1))
            y_pred_elem = reconstructed_model.predict(elem)
            # concate id and prediction
            id_pred_np = np.concatenate((ids_np_1dim, y_pred_elem), axis=1)
            # accumulate data
            all_id_pred = np.concatenate((all_id_pred, id_pred_np), axis=0)
        np.savetxt('test_pred.txt', all_id_pred, delimiter='\t', fmt='%s')
        print('done inference')

main_fun('train')
