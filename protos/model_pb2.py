# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='model.proto',
  package='bero.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n\x0bmodel.proto\x12\x0b\x62\x65ro.protos\"\xc6\x01\n\x0bModelConfig\x12\x1e\n\x08\x62\x61\x63kbone\x18\x01 \x01(\t:\x0cinception_v3\x12\x13\n\x06height\x18\x02 \x01(\r:\x03\x32\x39\x39\x12\x12\n\x05width\x18\x03 \x01(\r:\x03\x32\x39\x39\x12\x1a\n\tloss_type\x18\x04 \x01(\t:\x07softmax\x12\x13\n\x0bnum_classes\x18\x05 \x02(\r\x12\x1a\n\x0b\x61utoaugment\x18\x06 \x01(\x08:\x05\x66\x61lse\x12!\n\tmodeltype\x18\x07 \x01(\t:\x0e\x63lassification'
)




_MODELCONFIG = _descriptor.Descriptor(
  name='ModelConfig',
  full_name='bero.protos.ModelConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='backbone', full_name='bero.protos.ModelConfig.backbone', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"inception_v3".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='bero.protos.ModelConfig.height', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=299,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='bero.protos.ModelConfig.width', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=299,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_type', full_name='bero.protos.ModelConfig.loss_type', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"softmax".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='bero.protos.ModelConfig.num_classes', index=4,
      number=5, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='autoaugment', full_name='bero.protos.ModelConfig.autoaugment', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='modeltype', full_name='bero.protos.ModelConfig.modeltype', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"base".decode('utf-8'),
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
  serialized_start=29,
  serialized_end=227,
)

DESCRIPTOR.message_types_by_name['ModelConfig'] = _MODELCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelConfig = _reflection.GeneratedProtocolMessageType('ModelConfig', (_message.Message,), {
  'DESCRIPTOR' : _MODELCONFIG,
  '__module__' : 'model_pb2'
  # @@protoc_insertion_point(class_scope:bero.protos.ModelConfig)
  })
_sym_db.RegisterMessage(ModelConfig)


# @@protoc_insertion_point(module_scope)
