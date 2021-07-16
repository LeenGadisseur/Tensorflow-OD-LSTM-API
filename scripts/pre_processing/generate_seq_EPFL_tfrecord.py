# gebaseerd op script van petinhoss7 https://github.com/tensorflow/models/issues/7967


import os
import io
import glob
import math
import hashlib
import logging
import object_detection.utils.dataset_util as dataset_util
import tensorflow.compat.v1 as tf
import random

from lxml import etree
from PIL import Image



#EPFL class dict
class_dict = {'person': 1 }

flags = tf.app.flags
flags.DEFINE_string('root_dir', '/media/leen/Acer_500GB_HDD/EPFL', 'Root directory to raw EPFL dataset.')
flags.DEFINE_string('set', 'test', 'Convert training set, validation set.')
flags.DEFINE_string('output_path', 'Data/EPFL', 'Path to output TFRecord')
flags.DEFINE_integer('start_shard', 0, 'Start index of TFRcord files')
flags.DEFINE_integer('num_shards', 1, 'The number of TFRcord files')
flags.DEFINE_integer('num_frames', 10, 'The number of frame to use')
flags.DEFINE_integer('num_examples', -1, 'The number of video to convert to TFRecord file')
FLAGS = flags.FLAGS

SHUFFLED = False

SETS = ['train', 'val', 'test']
MAX_INTERVAL = 5

def sample_frames(xml_files):
	#print("Lengte XML files: ", len(xml_files))
	samples_size = (len(xml_files) - 1) // FLAGS.num_frames + 1
	print("Lengte sample size: ",samples_size)
	samples = []
	for s in range(samples_size):
		start = FLAGS.num_frames * s
		end = FLAGS.num_frames * (s+1)
		sample = xml_files[start:end]
		if s ==1000:
			print("Sample: ", sample)
		samples.append(sample)
	print("Lengte samples size: ", len(samples))
	return samples

def gen_shard(examples_list, annotations_dir, out_filename, root_dir, _set):
	print("Output file path : ", out_filename)
	writer = tf.python_io.TFRecordWriter(out_filename)
	#print("Example list: ",examples_list)
	print("Processing XML files...")
	xml_files = []
	print("Lengte examples_list: ", len(examples_list))

	for indx, example in enumerate(examples_list):
		#Groepeer files per 10
		#print("Index: ", indx)
		xml_file = os.path.join(annotations_dir, example + '.xml')
		xml_files.append(xml_file) 
	#print("Lengte xml_files: ", len(xml_files))
	samples = sample_frames(xml_files)
	#print("Lengte Samples: ", len(samples))
	
	#Al de samples die in 1 shard zitten
	for sample in samples:
		dicts = []
		#print("Sample: ", sample)
		for xml_file in sample:
		## process per single xml
			with tf.gfile.GFile(xml_file, 'r') as fid:
				xml_str = fid.read()
				xml = etree.fromstring(xml_str)
				dic = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
				dicts.append(dic)
		#for d in dicts:
			#print("fn d in dicts: ", d.get('filename'))
		tf_example = dicts_to_tf_example(dicts, root_dir, _set)
		#print(tf_example)
		writer.write(tf_example.SerializeToString())
	writer.close()


		#print("XML pattern: ", xml_pattern)
		
		
	#print("Dicts: ", dicts)
	
	return

def dicts_to_tf_example(dicts, root_dir, _set):
	""" Convert XML derived dict to tf.Example proto."""
	# Non sequential data
	folder = dicts[0]['folder']
	filenames = [dic['filename'] for dic in dicts]
	height = int(dicts[0]['size']['height'])
	width = int(dicts[0]['size']['width'])

	# Get image paths
	imgs_dir = os.path.join(root_dir, 'Data/{}'.format(_set), folder)
	print("imgs_dir: ", imgs_dir)
	imgs_path = sorted([os.path.join(imgs_dir, filename) + '.JPEG' for filename in filenames])
	print("imgs_path: ", imgs_path)

	# Frames Info (image)
	filenames = []
	encodeds = []
	sources = []
	keys = []
	formats = []
	is_annotateds = []
	# Frames Info (objects)
	xmins, ymins = [], []
	xmaxs, ymaxs = [], []
	class_indices = []
	names = []
	occludeds = []
	generateds = []

	# Iterate frames
	for data, img_path in zip(dicts, imgs_path):
		## open single frame
		with tf.gfile.FastGFile(img_path, 'rb') as fid:
			encoded_jpg = fid.read()
		encoded_jpg_io = io.BytesIO(encoded_jpg)
		image = Image.open(encoded_jpg_io)
		if image.format != 'JPEG':
			raise ValueError('Image format not JPEG')
		key = hashlib.sha256(encoded_jpg).hexdigest()
		is_annotated = 1 #Steeds 1 want alles is geannoteerd in deze dataset

	    ## validation
		assert int(data['size']['height']) == height
		assert int(data['size']['width']) == width

	    ## iterate objects
		xmin, ymin = [], []
		xmax, ymax = [], []
		class_index = []
		name = []
		occluded = []
		generated = []
		
		#geen_obj = False
		if 'object' in data:
			for obj in data['object']:
				xmin.append(float(obj['bndbox']['xmin']) / width)
				ymin.append(float(obj['bndbox']['ymin']) / height)
				xmax.append(float(obj['bndbox']['xmax']) / width)
				ymax.append(float(obj['bndbox']['ymax']) / height)
				#class_index.append(class_dict[obj['name']])
				class_index.append(class_dict.get(obj['name']))
				name.append(obj['name'].encode('utf8'))
				occluded.append(int(obj['occluded']))
				generated.append(int(obj['generated']))
				
			#print("class index: ", class_index)
		"""
		else:
			xmin.append(float(-1))
			ymin.append(float(-1))
			xmax.append(float(-1))
			ymax.append(float(-1))
			class_index.append(0)
			name.append('NoObject'.encode('utf8'))
			occluded.append(0)
			generated.append(0)
			print("Geen objecten in image: ")
			geen_obj = True"""
		
		
		## append tf_feature to list
		filenames.append(dataset_util.bytes_feature(data['filename'].encode('utf8')))
		encodeds.append(dataset_util.bytes_feature(encoded_jpg))
		source = os.path.join('EPFL/Data/{}'.format(_set), folder, data['filename'])
		sources.append(dataset_util.bytes_feature(source.encode('utf8')))
		keys.append(dataset_util.bytes_feature(key.encode('utf8')))
		is_annotateds.append(dataset_util.int64_feature(is_annotated))
		formats.append(dataset_util.bytes_feature('jpeg'.encode('utf8')))


		xmins.append(dataset_util.float_list_feature(xmin))
		ymins.append(dataset_util.float_list_feature(ymin))
		xmaxs.append(dataset_util.float_list_feature(xmax))
		ymaxs.append(dataset_util.float_list_feature(ymax))
		class_indices.append(dataset_util.int64_list_feature(class_index))
		names.append(dataset_util.bytes_list_feature(name))
		occludeds.append(dataset_util.int64_list_feature(occluded))
		generateds.append(dataset_util.int64_list_feature(generated))
		
		"""if(geen_obj==True):
			print("Filenames: ", filenames)		
			print("BB coord: ", xmins,ymins, xmaxs,ymaxs )
			print("class_indices: ", class_indices)
			print("label: ", names)
			geen_obj =False"""

	# Non sequential features
	context = tf.train.Features(feature={
		'image/folder': dataset_util.bytes_feature(folder.encode('utf8')),
		'image/frame_number': dataset_util.int64_feature(len(imgs_path)),
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		})
	# Sequential features
	tf_feature_lists = {
		'image/format': tf.train.FeatureList(feature=formats),
		'image/source_id': tf.train.FeatureList(feature=sources),
		'image/filename': tf.train.FeatureList(feature=filenames),
		'image/key/sha256': tf.train.FeatureList(feature=keys),
		'image/encoded': tf.train.FeatureList(feature=encodeds),
		'region/bbox/xmin': tf.train.FeatureList(feature=xmins),
		'region/bbox/xmax': tf.train.FeatureList(feature=xmaxs),
		'region/bbox/ymin': tf.train.FeatureList(feature=ymins),
		'region/bbox/ymax': tf.train.FeatureList(feature=ymaxs),
		'region/label/index': tf.train.FeatureList(feature=class_indices),
		'region/label/string': tf.train.FeatureList(feature=names),
		'region/occluded': tf.train.FeatureList(feature=occludeds),
		'region/generated': tf.train.FeatureList(feature=generateds),
		'region/is_annotated': tf.train.FeatureList(feature=is_annotateds),
		}
	feature_lists = tf.train.FeatureLists(feature_list=tf_feature_lists)
	#print("feature_lists: ", feature_lists)
	# Make single sequence example
	tf_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

	return tf_example

def EPFL_read_examples_list(path, examples_list):
	with tf.gfile.GFile(path) as fid:
		lines = fid.readlines()
	for line in lines:
		l_10 = line.strip().split(',')[:10]
		examples_list.extend(l_10)
	return

def EPFL_read_examples_shuffled_list(path, examples_list):
	with tf.gfile.GFile(path) as fid:
		lines = fid.readlines()
	lijst = [line.strip().split(',')[:10] for line in lines]
	#2D lijst shuffle
	#print("Voor shuffle: ", lijst[:3])
	random.shuffle(lijst)
	#print("Na shuffle: ", lijst[:3])
	
	#Geshufflede 2D lijst omzetten naar 1D lijst
	flat = [l for l10 in lijst for l in l10]
	#print("1D lijt: ", flat[:30])
	examples_list.extend(flat)
	
	return 
	 

def main():
	root_dir = FLAGS.root_dir

	if FLAGS.set not in SETS:
	    raise ValueError('set must be in : {}'.format(SETS))

	# Read Example list files
	logging.info('Reading from EPFL dataset. ({})'.format(root_dir))
	list_file_pattern = 'ImageSets/{}*.txt'.format(FLAGS.set)
	print("List file patterns: ", list_file_pattern )

	#Bij gebruik van meerdere .txt files
	examples_paths = sorted(glob.glob(os.path.join(root_dir, list_file_pattern)))
	#print('examples_paths', examples_paths)
	examples_list = []
	for examples_path in examples_paths:
		print("Example path: ", examples_path)
		#examples_list.extend(dataset_util.read_examples_list(examples_path))# voor VID dataset, splitst op spaties
		EPFL_read_examples_shuffled_list(examples_path, examples_list)
		print("example list: ", examples_list[:30])
	if FLAGS.set != 'train':
		#examples_list2 = [e[:-7] for e in examples_list] #TO DO: checken wat dit doet
		examples_list = sorted(list(set(examples_list)))
		#print("Example list: ", examples_list)
		print("Geen training set.")
	if FLAGS.num_examples > 0:
		examples_list = examples_list[:FLAGS.num_examples]
		print("Flag num_examples.")
	#print('examples_list', examples_list)

	# Sharding
	start_shard = FLAGS.start_shard
	num_shards = FLAGS.num_shards
	num_digits = math.ceil(math.log10(max(num_shards-1,2)))
	shard_format = '%0'+ ('%d'%num_digits) + 'd'
	examples_per_shard = int(math.ceil(len(examples_list)/float(num_shards)))
	#print("Examples per shard: ", examples_per_shard)
	annotations_dir = os.path.join(root_dir,
		                       'Annotations/{}'.format(FLAGS.set))
	print('annotations_dir', annotations_dir)
	# Generate each shard
	for i in range(start_shard, num_shards):
		start = i * examples_per_shard
		end = (i+1) * examples_per_shard
		out_filename = os.path.join(FLAGS.output_path,
		    FLAGS.set+'-EPFL-'+(shard_format % i)+'.record')
		if os.path.isfile(out_filename): # Don't recreate data if restarting
			continue
		print ("Shard: "+ str(i) +' of '+ str(num_shards) + " shards " +' ['+str(start)+':'+str(end),'] '+ "Outfile: " +out_filename)
		gen_shard(examples_list[start:end], annotations_dir, out_filename,
		    root_dir, FLAGS.set)
		#print(examples_list[start:end])
	return

main()
tf.app.run()
