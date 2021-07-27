import tensorflow.compat.v1 as tf 



filenames = ['./test_records/snapshot_serengeti_sequence_examples.record','test-EPFL-0-4fr.record','shards/train-EPFL-00.record']
raw_dataset = tf.data.TFRecordDataset(filenames[2])

for raw_record in raw_dataset.take(100):
   
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
    example_seq = tf.train.SequenceExample()
    example_seq.ParseFromString(raw_record.numpy())
    print(example_seq)
