# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lstm_object_detection/protos/quant_overrides.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lstm_object_detection/protos/quant_overrides.proto',
  package='lstm_object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n2lstm_object_detection/protos/quant_overrides.proto\x12\x1clstm_object_detection.protos\"R\n\x0eQuantOverrides\x12@\n\rquant_configs\x18\x01 \x03(\x0b\x32).lstm_object_detection.protos.QuantConfig\"\xbd\x01\n\x0bQuantConfig\x12\x0f\n\x07op_name\x18\x01 \x02(\t\x12\x15\n\rquant_op_name\x18\x02 \x02(\t\x12\x1a\n\x0b\x66ixed_range\x18\x03 \x02(\x08:\x05\x66\x61lse\x12\x0f\n\x03min\x18\x04 \x01(\x02:\x02-6\x12\x0e\n\x03max\x18\x05 \x01(\x02:\x01\x36\x12\x15\n\x05\x64\x65lay\x18\x06 \x01(\x05:\x06\x35\x30\x30\x30\x30\x30\x12\x16\n\x0bweight_bits\x18\x07 \x01(\x05:\x01\x38\x12\x1a\n\x0f\x61\x63tivation_bits\x18\x08 \x01(\x05:\x01\x38')
)




_QUANTOVERRIDES = _descriptor.Descriptor(
  name='QuantOverrides',
  full_name='lstm_object_detection.protos.QuantOverrides',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='quant_configs', full_name='lstm_object_detection.protos.QuantOverrides.quant_configs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=166,
)


_QUANTCONFIG = _descriptor.Descriptor(
  name='QuantConfig',
  full_name='lstm_object_detection.protos.QuantConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='op_name', full_name='lstm_object_detection.protos.QuantConfig.op_name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quant_op_name', full_name='lstm_object_detection.protos.QuantConfig.quant_op_name', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fixed_range', full_name='lstm_object_detection.protos.QuantConfig.fixed_range', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min', full_name='lstm_object_detection.protos.QuantConfig.min', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-6),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max', full_name='lstm_object_detection.protos.QuantConfig.max', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(6),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delay', full_name='lstm_object_detection.protos.QuantConfig.delay', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=500000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_bits', full_name='lstm_object_detection.protos.QuantConfig.weight_bits', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation_bits', full_name='lstm_object_detection.protos.QuantConfig.activation_bits', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=169,
  serialized_end=358,
)

_QUANTOVERRIDES.fields_by_name['quant_configs'].message_type = _QUANTCONFIG
DESCRIPTOR.message_types_by_name['QuantOverrides'] = _QUANTOVERRIDES
DESCRIPTOR.message_types_by_name['QuantConfig'] = _QUANTCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QuantOverrides = _reflection.GeneratedProtocolMessageType('QuantOverrides', (_message.Message,), dict(
  DESCRIPTOR = _QUANTOVERRIDES,
  __module__ = 'lstm_object_detection.protos.quant_overrides_pb2'
  # @@protoc_insertion_point(class_scope:lstm_object_detection.protos.QuantOverrides)
  ))
_sym_db.RegisterMessage(QuantOverrides)

QuantConfig = _reflection.GeneratedProtocolMessageType('QuantConfig', (_message.Message,), dict(
  DESCRIPTOR = _QUANTCONFIG,
  __module__ = 'lstm_object_detection.protos.quant_overrides_pb2'
  # @@protoc_insertion_point(class_scope:lstm_object_detection.protos.QuantConfig)
  ))
_sym_db.RegisterMessage(QuantConfig)


# @@protoc_insertion_point(module_scope)
