# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: train.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='train.proto',
  package='bero.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x0btrain.proto\x12\x0b\x62\x65ro.protos\"\xf9\x01\n\x0bTrainConfig\x12\x16\n\nbatch_size\x18\x01 \x01(\r:\x02\x36\x34\x12\x1c\n\x14\x66ine_tune_checkpoint\x18\x02 \x01(\t\x12\x16\n\nmax_epochs\x18\x03 \x01(\r:\x02\x31\x30\x12#\n\x15initial_learning_rate\x18\x04 \x01(\x02:\x04\x30.01\x12\x15\n\toptimizer\x18\x05 \x01(\t:\x02gd\x12\x11\n\ttrain_dir\x18\x06 \x02(\t\x12\x18\n\nload_model\x18\x07 \x01(\x08:\x04true\x12\x10\n\x08\x64\x61ta_dir\x18\x08 \x01(\t\x12\x0c\n\x04gpus\x18\t \x03(\r\x12\x13\n\x0bvaldata_dir\x18\n \x01(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='bero.protos.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='bero.protos.TrainConfig.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fine_tune_checkpoint', full_name='bero.protos.TrainConfig.fine_tune_checkpoint', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_epochs', full_name='bero.protos.TrainConfig.max_epochs', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initial_learning_rate', full_name='bero.protos.TrainConfig.initial_learning_rate', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.01),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='optimizer', full_name='bero.protos.TrainConfig.optimizer', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("gd").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='train_dir', full_name='bero.protos.TrainConfig.train_dir', index=5,
      number=6, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='load_model', full_name='bero.protos.TrainConfig.load_model', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data_dir', full_name='bero.protos.TrainConfig.data_dir', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gpus', full_name='bero.protos.TrainConfig.gpus', index=8,
      number=9, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='valdata_dir', full_name='bero.protos.TrainConfig.valdata_dir', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=278,
)

DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINCONFIG,
  __module__ = 'train_pb2'
  # @@protoc_insertion_point(class_scope:bero.protos.TrainConfig)
  ))
_sym_db.RegisterMessage(TrainConfig)


# @@protoc_insertion_point(module_scope)
