from __future__ import print_function
import sys, os
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
import conv_base
import ssd_base
#from conv_base import construct_net_body
#from ssd_base import CreateAnnotatedDataLayer
#from ssd_base import CreateMultiBoxHead

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_net(net_paras, all_cfgs):
  batch_sampler, trp, tep,  mlp, lp, pbp, sp, dop, dep = ssd_base.set_all_cfgs(all_cfgs)
  net_cfgs = all_cfgs['net_cfgs']

  def make_train_net():
    net = caffe.NetSpec()
    net.data, net.label = ssd_base.CreateAnnotatedDataLayer(net_cfgs['train_data'], batch_size=net_cfgs['batch_size'],
          train=True, output_label=True, label_map_file=net_cfgs['label_map_file'],
          transform_param=trp, batch_sampler=batch_sampler)

    net = conv_base.construct_net_body(net, *net_paras)
    mbox_layers = ssd_base.CreateMultiBoxHead(net, data_layer='data', from_layers=pbp['mbox_source_layers'],
          use_batchnorm=False, min_sizes=pbp['min_sizes'], max_sizes=pbp['max_sizes'],
          aspect_ratios=pbp['aspect_ratios'], steps=pbp['steps'], normalizations=pbp['normalizations'],
          num_classes=all_cfgs['multibox_cfgs']['num_classes'], share_location=mlp['share_location'], flip=pbp['flip'], 
	  clip=pbp['clip'], prior_variance=pbp['prior_variance'], kernel_size=3, pad=1, lr_mult=1.0)
  
    ## Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=mlp,
          loss_param=lp, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
          propagate_down=[True, True, False, False])
    with open(net_cfgs['train_file_path'], 'w') as f:
      print('name: "{}_train"'.format(net_cfgs['model_name']), file=f)
      print(net.to_proto(), file=f) 
    return

  def make_test_net():
    net = caffe.NetSpec()
    net.data, net.label = ssd_base.CreateAnnotatedDataLayer(net_cfgs['test_data'], batch_size=net_cfgs['batch_size'],
          train=False, output_label=True, label_map_file=net_cfgs['label_map_file'],
          transform_param=tep)

    net = conv_base.construct_net_body(net, *net_paras)
    mbox_layers = ssd_base.CreateMultiBoxHead(net, data_layer='data', from_layers=pbp['mbox_source_layers'],
          use_batchnorm=False, min_sizes=pbp['min_sizes'], max_sizes=pbp['max_sizes'],
          aspect_ratios=pbp['aspect_ratios'], steps=pbp['steps'], normalizations=pbp['normalizations'],
          num_classes=all_cfgs['multibox_cfgs']['num_classes'], share_location=mlp['share_location'], flip=pbp['flip'], 
	  clip=pbp['clip'], prior_variance=pbp['prior_variance'], kernel_size=3, pad=1, lr_mult=1.0)
    conf_name = "mbox_conf"
    reshape_name = "{}_reshape".format(conf_name)
    net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, all_cfgs['multibox_cfgs']['num_classes']]))
    softmax_name = "{}_softmax".format(conf_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(conf_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[1] = net[flatten_name]
    
    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=dop,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=dep,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(net_cfgs['test_file_path'], 'w') as f:
      print('name: "{}_test"'.format(net_cfgs['model_name']), file=f)
      print(net.to_proto(), file=f) 
    return net

  def make_deploy_net():
    deploy_net = make_test_net()
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(net_cfgs['model_name'])
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, all_cfgs['transform_cfgs']['height'], all_cfgs['transform_cfgs']['width']])])
    with open(net_cfgs['deploy_file_path'], 'w') as f:
      print(net_param, file=f)

  def make_solver():
    solver = caffe_pb2.SolverParameter(
                  train_net=net_cfgs['train_file_path'],
                  test_net=[net_cfgs['test_file_path']],
                  snapshot_prefix=net_cfgs['snapshot_prefix'], **sp)

    with open(net_cfgs['solver_file_path'], 'w') as f:
      print(solver, file=f)

  def make_job():
    train_src_param = '--weights="{}" \\\n'.format(net_cfgs['pretrain_model']) 
    with open(net_cfgs['job_file_path'], 'w') as f:
      caffe_bin_path = os.path.join(all_cfgs['job_cfgs']['caffe_root'], '/build/tools/caffe')
      f.write('{} train \\\n'.format(caffe_bin_path))
      f.write('--solver="{}" \\\n'.format(net_cfgs['solver_file_path']))
      f.write(train_src_param)
      f.write('--gpu 0 2>&1 |tee {}'.format(net_cfgs['log_file_path']))

       

  #make_solver()
  make_train_net()
  make_deploy_net()
  make_solver()
  make_job()

  return 

def check_job_path(job_cfgs, net_cfgs):
  check_if_exist(net_cfgs['train_data'])
  check_if_exist(net_cfgs['test_data'])
  check_if_exist(net_cfgs['label_map_file'])
  check_if_exist(net_cfgs['pretrain_model'])
  make_if_not_exist(job_cfgs['job_dir'])
  make_if_not_exist(job_cfgs['snapshot_dir'])
  make_if_not_exist(job_cfgs['output_dir'])
  net_cfgs['train_file_path'] = os.path.join(job_cfgs['job_dir'], "train.prototxt")
  net_cfgs['test_file_path'] = os.path.join(job_cfgs['job_dir'], "test.prototxt")
  net_cfgs['deploy_file_path'] = os.path.join(job_cfgs['job_dir'], "deploy.prototxt")
  net_cfgs['solver_file_path'] = os.path.join(job_cfgs['job_dir'], "solver.prototxt")
  net_cfgs['job_file_path'] = os.path.join(job_cfgs['job_dir'], "train.sh")
  net_cfgs['snapshot_prefix'] = os.path.join(job_cfgs['snapshot_dir'], net_cfgs['model_name'])
  net_cfgs['log_file_path'] = os.path.join(job_cfgs['job_dir'], net_cfgs['model_name']+".log")


if __name__ == '__main__':
  net_yaml_path = sys.argv[1]
  cfg_yaml_path = sys.argv[2]
  import yaml
  with open(net_yaml_path, 'r') as f:
    net_paras = yaml.load(f)
  with open(cfg_yaml_path, 'r') as f:
    all_cfgs = yaml.load(f)
  check_job_path(all_cfgs['job_cfgs'], all_cfgs['net_cfgs'])
  make_net(net_paras, all_cfgs)
  #print net_str
  

#temp =  example_network()f
#print temp
