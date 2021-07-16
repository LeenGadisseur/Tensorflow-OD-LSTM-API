import tensorflow.compat.v2 as tf 



filenames = ['./test_records/snapshot_serengeti_sequence_examples.record','test-EPFL-0.record','train-EPFL-0.record']
raw_dataset = tf.data.TFRecordDataset(filenames[0])

for raw_record in raw_dataset.take(100):
   
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
    example_seq = tf.train.SequenceExample()
    example_seq.ParseFromString(raw_record.numpy())
    print(example_seq)
