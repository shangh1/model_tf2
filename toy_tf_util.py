from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np

feature_delimiter = '\t'

import tensorflow_text
import tensorflow as tf
import collections

# THIS IS NOW COMPACTABLE WITH TF2.x
#tf.enable_eager_execution()

def merge_dims(rt, axis=0):
    ## https://github.com/tensorflow/text/issues/155
    to_expand = rt.nested_row_lengths()[axis]
    to_elim = rt.nested_row_lengths()[axis + 1]

    bar = tf.RaggedTensor.from_row_lengths(to_elim, row_lengths=to_expand)
    new_row_lengths = tf.reduce_sum(bar, axis=1)
    return tf.RaggedTensor.from_nested_row_lengths(
        rt.flat_values,
        rt.nested_row_lengths()[:axis] + (new_row_lengths,))

def getTokenFromVocab(vocabTableObject,sentenceQuery):
    tokenizerWhiteSpace = tensorflow_text.WhitespaceTokenizer()
    tokens = tokenizerWhiteSpace.tokenize(sentenceQuery)
    tokenizer = tensorflow_text.WordpieceTokenizer(vocabTableObject, token_out_type=tf.string)
    result = tokenizer.tokenize(tokens)
    return result

def tokenizationCustomized(vocabTableObject,sentence,segment_value):

    result = getTokenFromVocab(vocabTableObject,sentence)
    result = merge_dims(result,0)

    numOfExamplesBefore = result.nrows()
    sepTensor = tf.fill([numOfExamplesBefore,1],"[SEP]")
    clippedAdded = tf.concat([result, sepTensor], axis=1)

    where_True = tf.not_equal(clippedAdded, 'eiddccidrh') ### 'eiddccidrh' is random string
    where_int = tf.cast(where_True, tf.int32)
    segment_ids = where_int * segment_value
    return clippedAdded, segment_ids

def tableLookUp(vocabTableObject,sentence):
    ids = vocabTableObject.lookup(sentence)
    return ids

def convertTokensToIdsWithPadding(vocabTableObject,tokensTensor, seq_length_allowed):

    numOfExamplesBefore = tokensTensor.nrows()
    clsTensor = tf.fill([numOfExamplesBefore,1],"[CLS]")
    clippedAdded = tf.concat([clsTensor, tokensTensor], axis=1)
    clippedPadded = tf.concat([clippedAdded, [[''] * (seq_length_allowed)]], axis=0)
    clippedPaddedTensor = clippedPadded.to_tensor()

    ### removing last row
    clippedPaddedTensor = clippedPaddedTensor[:-1,:seq_length_allowed]
    input_ids = tableLookUp(vocabTableObject,clippedPaddedTensor)
    input_ids = tf.cast(input_ids,dtype=tf.int32)
    #### limiting to seq_length_allowed allowed
    return input_ids[:,:seq_length_allowed]

def lowerText(sentence):
    lower = tf.strings.lower(sentence)
    return lower

def regexReplace(sentence,pattern):
    cleaned = tf.strings.regex_replace(sentence, pattern, '')
    return cleaned

def cleanText(textRaw):
    lowertextRaw = lowerText(textRaw)
    cleaned = regexReplace(lowertextRaw,"[^a-zA-Z0-9 ''$%]+")
    return cleaned

def splitFL(inputString, num_class):
    split_result = tf.strings.split(inputString, sep='\t')
    split_result_tensor = split_result.to_tensor()
    label_indices = split_result_tensor[:, 2]
    each_label = tf.strings.split(label_indices, sep=" ").to_sparse()
    indices_label = each_label.indices[:,0]
    values_label = each_label.values
    values_label = tf.strings.to_number(values_label, tf.dtypes.int64)
    new_indices_label = tf.stack([indices_label, values_label],1)
    ones_label  = tf.ones(shape=(tf.shape(input=new_indices_label)[0],), dtype=tf.float32)
    sparse= tf.sparse.SparseTensor(indices=new_indices_label, values=ones_label, dense_shape=[tf.shape(input=label_indices)[0],num_class])
    batch_labels = tf.compat.v1.sparse_to_dense(sparse.indices,sparse.dense_shape,sparse.values)

    return split_result_tensor, batch_labels


def MecKaminoFeature(inputString,vocab_file_path, seq_length,num_class):
    ### where id field starts
    num_id_fields = 3 # 3 for MEC
    vocab_table_object =  tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(vocab_file_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), 0)

    split_result = tf.strings.split(inputString, sep='\t')
    split_result_tensor = split_result.to_tensor()

    ##### batch_ids START #####
    batch_ids = split_result_tensor[:,:num_id_fields+1]
    ##### batch_ids END #####

    ###### batch_labels start #######
    label_indices = split_result_tensor[:,num_id_fields-1] #['0 23']
    each_label = tf.strings.split(label_indices, sep=" ").to_sparse()
    indices_label = each_label.indices[:,0]
    values_label = each_label.values
    values_label = tf.strings.to_number(values_label, tf.dtypes.int64)
    new_indices_label = tf.stack([indices_label, values_label],1)
    ones_label  = tf.ones(shape=(tf.shape(input=new_indices_label)[0],), dtype=tf.float32)
    sparse= tf.sparse.SparseTensor(indices=new_indices_label, values=ones_label, dense_shape=[tf.shape(input=label_indices)[0],num_class])
    batch_labels = tf.compat.v1.sparse_to_dense(sparse.indices,sparse.dense_shape,sparse.values)
    ###### batch_labels end #######

    ##### batch_input_ids #####
    sender = split_result_tensor[:,num_id_fields]
    subject = split_result_tensor[:,num_id_fields+4] ### change for MEC -4

    sender = cleanText(sender)
    subject = cleanText(subject)

    sender_tokens, sender_segment_ids = tokenizationCustomized(vocab_table_object, sender, 0)
    subject_tokens, subject_segment_ids = tokenizationCustomized(vocab_table_object, subject, 1) #### FOR MEC

    # ADD ADDITIONAL "SEP" for xpath
    numOfExamplesBefore = subject_tokens.nrows()
    sepTensor = tf.fill([numOfExamplesBefore,1],"[SEP]")
    subject_tokens_sepAdded = tf.concat([subject_tokens, sepTensor], axis=1)

    #############
    short_seq = tf.concat([sender_tokens, subject_tokens_sepAdded], 1) ### change for MEC
    short_seq = short_seq[:,:seq_length-1]
    batch_input_ids = convertTokensToIdsWithPadding(vocab_table_object,short_seq, seq_length)

    return batch_ids, batch_labels,batch_input_ids