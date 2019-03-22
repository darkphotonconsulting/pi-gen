#!/usr/bin/env python3

import argparse
import ffmpeg as avengine
import mxnet as mx
import numpy as np
import cv2
import datetime as dt
import boto3 as aws
import botocore
import time
import os
import sys
import json
from botocore.exceptions import ClientError
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

parser = argparse.ArgumentParser(description='ML Pi Image Classification Tool')


parser.add_argument('--list_nets', help='List Usable Model Zoo Networks', action='store_true', required=False, default=False)
parser.add_argument('--snapshot', help='Run Classification Against Web Camera Input', action='store_true', required=False, default=False)
parser.add_argument('--local', help='Run Classification Against Local File', action='store_true', required=False, default=False)
parser.add_argument('--remote', help='Run Classification Against Remote File (URL)', action='store_true', required=False, default=False)
parser.add_argument('--camera_device', help='Video 4 Linux Device Path', action='store', required=False, default='/dev/video0')
parser.add_argument('--img_file', help='Path To Local Image File', action='store', required=False, default='/tmp/example.jpg')
parser.add_argument('--img_url', help='URL To Remote Image File', action='store', required=False, default='https://vetstreet-brightspot.s3.amazonaws.com/f6/df2f40a33911e087a80050568d634f/file/Egyptian-Mau-4-645mk062311.jpg')
parser.add_argument('--s3_bucket', help='Target AWS S3 Bucket For Uploads', action='store', required=False, default='darkphotonconsultingllc-mlpi')
parser.add_argument('--enable_upload', help='Target AWS S3 Bucket For Uploads', action='store_true', required=False, default=False)
parser.add_argument('--zoo_network', help='Target AWS S3 Bucket For Uploads', action='store', required=False, default='resnet-18')

MODEL_ZOO = {
  "base": "http://data.mxnet.io/models/imagenet/" ,
  "caffenet": {
    "params": "caffenet/caffenet-0000.params",
    "symbol": "caffenet/caffenet-symbol.json",
    "synset": "synset.txt"
  },
  "inception": {
    "params": "inception-bn/Inception-BN-0126.params",
    "symbol": "inception-bn/Inception-BN-symbol.json",
    "synset": "synset.txt"
  },
  "nin": {
    "params": "nin/nin-0000.params",
    "symbol": "nin/nin-symbol.json",
    "synset": "synset.txt"
  },
  "nin": {
    "params": "nin/nin-0000.params",
    "symbol": "nin/nin-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-18": {
    "params": "resnet/18-layers/resnet-18-0000.params",
    "symbol": "resnet/18-layers/resnet-18-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-34": {
    "params": "resnet/34-layers/resnet-34-0000.params",
    "symbol": "resnet/34-layers/resnet-34-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-50": {
    "params": "resnet/50-layers/resnet-50-0000.params",
    "symbol": "resnet/50-layers/resnet-50-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-101": {
    "params": "resnet/101-layers/resnet-101-0000.params",
    "symbol": "resnet/101-layers/resnet-101-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-152": {
    "params": "resnet/152-layers/resnet-152-0000.params",
    "symbol": "resnet/152-layers/resnet-152-symbol.json",
    "synset": "synset.txt"
  },
  "resnet-200": {
    "params": "resnet/200-layers/resnet-200-0000.params",
    "symbol": "resnet/200-layers/resnet-200-symbol.json",
    "synset": "synset.txt"
  },
  "resnext-50": {
    "params": "resnext/50-layers/resnext-50-0000.params",
    "symbol": "resnext/50-layers/resnext-50-symbol.json",
    "synset": "synset.txt"
  },
  "resnext-101": {
    "params": "resnext/101-layers/resnext-101-0000.params",
    "symbol": "resnext/101-layers/resnext-101-symbol.json",
    "synset": "synset.txt"
  }
}

args = parser.parse_args()


#helper functions
def file_name(ospath='/tmp'):
  prfx = "mlpi"
  timestamp = dt.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
  data = "{}/{}-{}.jpg".format(ospath,prfx, timestamp)
  return data

def fetch_image(url,ospath='/tmp'):
  data = mx.test_utils.download(url, fname="{}/{}".format(ospath,url.split('/')[-1].split('?')[0]))
  return data

#AV functions
def take_picture(in_file, out_file):
  try:
    avengine.input(in_file).output(out_file, vframes=1).global_args('-loglevel','quiet').run(overwrite_output=True)
    data = out_file
    return data
  except avengine._run.Error as e:
    sys.exit("Unable To Use Input Device, Check If It Is In Use By Another Program")

def label_picture(in_file, label_data):
  label_data = "\n".join(label_data)
  out_file = in_file #overwrite
  avengine.input(in_file).drawtext(text=label_data, x=0,y=0, escape_text=True).output(out_file).global_args('-loglevel', 'quiet').run(overwrite_output=True)
  data = out_file
  return data

#AWS functions
def bucket_exists(bucket='darkphotonconsultingllc-mlpi'):
  client = aws.client('s3')
  try:
    client.get_bucket_location(Bucket=bucket)
    data = True
    return data
  except client.exceptions.NoSuchBucket as e:
    data = False
  except client.exceptions.ClientError as e:
    data = False
  return data

def create_bucket(bucket='darkphotonconsultingllc-mlpi'):
  client = aws.client('s3')
  try:
    client.create_bucket(
      ACL='private',
      Bucket=bucket,
      ObjectLockEnabledForBucket=False
    )
    data = bucket
  except client.exceptions.ClientError as e:
    print(e)
    data = False
  return data


def upload_file(in_file, bucket='darkphotonconsultingllc-mlpi'):
  client = aws.client('s3')
  out_file = os.path.basename(in_file)
  try:
    client.upload_file(in_file, bucket, out_file)
    data = out_file
  except botocore.exceptions.ClientError as e:
    data = False
  return data

#ML functions - Model Setup & Configuration
def set_context():
  data = mx.cpu()
  return data

def fetch_model_data():
  path='http://data.mxnet.io/models/imagenet/'
  data = [
    mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
    mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
    mx.test_utils.download(path+'synset.txt')
  ]
  return data

def fetch_model_data_v2(zoo_net,zoo_lib):
  path=zoo_lib['base']
  #path='http://data.mxnet.io/models/imagenet/'
  params = zoo_lib[zoo_net]['params']
  symbol = zoo_lib[zoo_net]['symbol']
  synset = zoo_lib[zoo_net]['synset']

  data = [
    mx.test_utils.download(path+params),
    mx.test_utils.download(path+symbol),
    mx.test_utils.download(path+synset)
  ]
  return data

def load_model_data():
  ctx = set_context()
  sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
  data = mx.mod.Module(
    symbol=sym,
    context=ctx,
    label_names=None
  )
  data.bind(
    for_training=False,
    data_shapes=[('data', (1,3,224,224))],
    label_shapes=data._label_shapes
  )
  data.set_params(arg_params, aux_params, allow_missing=True)
  return data

def load_model_data_v2(zoo_net):
  ctx = set_context()
  sym, arg_params, aux_params = mx.model.load_checkpoint(zoo_net, 0)
  data = mx.mod.Module(
    symbol=sym,
    context=ctx,
    label_names=None
  )
  data.bind(
    for_training=False,
    data_shapes=[('data', (1,3,224,224))],
    label_shapes=data._label_shapes
  )
  data.set_params(arg_params, aux_params, allow_missing=True)
  return data


def get_labels():
  with open('synset.txt', 'r') as f:
    data = [l.rstrip() for l in f]
  return data


#ML functions - prediction/feedback
#change convert image to use cv2 instead of loading image via MX which requires a rebuild of MXnet. 

def convert_image(in_file):
  #data = mx.image.imread(in_file)
  #data = mx.image.imresize(data, 224, 224)
  #data = data.transpose((2, 0, 1))
  #data = data.expand_dims(axis=0)
  #data = data.astype('float32')
  data = cv2.cvtColor(cv2.imread(in_file), cv2.COLOR_BGR2RGB)
  if data is None:
    return None
  data = cv2.resize(data, (224,224))
  data = np.swapaxes(data, 0, 2)
  data = np.swapaxes(data, 1, 2)
  data = data[np.newaxis, :]
  return data

# returns nothing currently for debugging, eventually want to build a clean string and return as `data`
def classify_image(model, labels, img):
  data = list()
  model.forward(Batch([mx.nd.array(img)]))
  try:
    prob = model.get_outputs()[0]
    prob = prob.asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    #print top 5
    for i in a[0:5]:
      data.append( "Class: {} Probability: {}".format(labels[i], prob[i]) )
      print('probability=%f, class=%s' %(prob[i], labels[i]))
    return data
  except mx.base.MXNetError as ex:
    sys.exit("MXNet error, bailing")

def main(a):
  #handle program flow flag ctrl parameters
  if a.list_nets:
    nets = list(MODEL_ZOO.keys())
    nets.sort()
    data = '\n'.join(nets[1:])
    print(data)
    sys.exit(0)
  elif a.snapshot:
    print("Taking Snapshot, Ensure Target Is Focused")
    img_file = take_picture(a.camera_device, file_name())
  elif a.local:
    img_file = a.img_file
    print("Using Local File {}".format(img_file))
  elif a.remote:
    img_file = fetch_image(a.img_url)
    print("Using Remote File {}".format(img_file))
  else:
    sys.exit("Please provide either --snapshot --local or --remote arguments")

  #model_data = fetch_model_data()
  model_data = fetch_model_data_v2(a.zoo_network,MODEL_ZOO)
  print("Downloaded Model Data {}".format(model_data))
  #model = load_model_data()
  model = load_model_data_v2(a.zoo_network)
  labels = get_labels()
  print("Loaded Model & {} Corresponding Labels".format(len(labels)))
  imgc = convert_image(img_file)
  print("Converted Source Image To Array Data")
  cdata = classify_image(model, labels, imgc)
  img_file = label_picture(img_file, cdata)
  if a.enable_upload:
    if bucket_exists(bucket=a.s3_bucket):
      f = upload_file(img_file, a.s3_bucket)
      print("Image Saved To s3://{}/{}".format(a.s3_bucket, f))
    else:
      create_bucket(bucket=a.s3_bucket)
      f = upload_file(img_file, a.s3_bucket)
      print("Image Saved To s3://{}/{}".format(a.s3_bucket, f))


main(args)

