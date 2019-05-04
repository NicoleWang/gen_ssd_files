import math
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import conv_base

#create ssd annotated_data_layer
def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)

## create mbox layers
def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}/mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        _, _, loc_conv_para = conv_base.conv_params(from_layer, name, num_loc_output, kernel_size)
        del loc_conv_para['bias_term']
	loc_conv_para['param'].append(dict(lr_mult=2, decay_mult=0))
	loc_conv_para['bias_filler'] = dict(type='constant',value=0)
	net = conv_base.Conv(net, from_layer, name, loc_conv_para)
        #ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
        #    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}/perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}/flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}/mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        _, _, conf_conv_para = conv_base.conv_params(from_layer, name, num_conf_output, kernel_size)
        del conf_conv_para['bias_term']
	conf_conv_para['param'].append(dict(lr_mult=2, decay_mult=0))
	conf_conv_para['bias_filler'] = dict(type='constant',value=0)
	net = conv_base.Conv(net, from_layer, name, conf_conv_para)
        #ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
        #    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}/perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}/flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}/mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

##  set batch_sampler
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]

#set train_transform_param
train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': 300,
                'width': 300,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }

## set test_transform_param 
test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': 300,
                'width': 300,
                'interp_mode': [P.Resize.LINEAR],
                },
        }

# MultiBoxLoss parameters.
#loc_weight = (neg_pos_ratio + 1.0) / 4.0
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': 1.0,
    'num_classes': 2,
    'share_location': True,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': 0,
    'use_difficult_gt': True,
    'mining_type': P.MultiBoxLoss.MAX_NEGATIVE,
    'neg_pos_ratio': 3,
    'neg_overlap': 0.5,
    'code_type': P.PriorBox.CENTER_SIZE,
    'ignore_cross_boundary_bbox': False,
    }

## set loss_param
loss_param = {
    'normalization': P.Loss.VALID,
    }

prior_box_param = {
	'min_dim': 300,
	'mbox_source_layers': ['conv_4', 'conv10'],
	'min_ratio': 20,
	'max_ratio': 90,
	'steps': [8,16],
	'aspect_ratios': [[2,2.5,3], [2,2.5,3]],
	'normalizations': [-1,-1],
	'prior_variance': [0.1,0.1,0.2,0.2],
	'flip': False,
	'clip': False
	}

def set_transform_cfgs(cfgs):
  for key, value in cfgs.items():
    train_transform_param['resize_param'][key] = value
    test_transform_param['resize_param'][key] = value

def set_multibox_cfgs(cfgs):
  for key, value in cfgs.items():
    multibox_loss_param[key] = value

def set_priorbox_cfgs(cfgs): 
  for key, value in cfgs.items():
    prior_box_param[key] = value
  if not ('min_sizes' in cfgs.keys()) or not('max_sizes' in cfgs.keys()):
    min_ratio = prior_box_param['min_ratio']
    max_ratio = prior_box_param['max_ratio']
    min_dim = prior_box_param['min_dim']
    mbox_source_layers = prior_box_param['mbox_source_layers']
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 1)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
      min_sizes.append(min_dim * ratio / 100.)
      max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [min_dim * 20 / 100.] + max_sizes
    prior_box_param['min_sizes'] = min_sizes
    prior_box_param['max_sizes'] = max_sizes   
     

solver_param = {
    # Train parameters
    'base_lr': 0.001,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [80000, 100000, 120000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': 1,
    'max_iter': 120000,
    'snapshot': 80000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': P.Solver.GPU,
    'device_id': 0,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [100],
    'test_interval': 10000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': 2,
    'share_location': True,
    'background_label_id': 0,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    #'save_output_param': {
    #    'output_directory': "",
    #    'output_name_prefix': "comp4_det_test_",
    #    'output_format': "VOC",
    #    'label_map_file': "",
    #    'name_size_file': "",
    #    'num_test_image': 0,
    #    },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': P.PriorBox.CENTER_SIZE,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': 2,
    'background_label_id': 0,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': "",
    }

def set_det_cfgs(cfgs):
  det_out_param['num_classes'] = multibox_loss_param['num_classes']
  det_out_param['background_label_id'] = multibox_loss_param['background_label_id']
  #det_out_param['save_output_param']['output_directory'] = cfgs['job_cfgs']['output_dir']
  #@det_out_param['save_output_param']['label_map_file'] = cfgs['net_cfgs']['label_map_file']
  #det_out_param['save_output_param']['name_size_file'] = cfgs['net_cfgs']['name_size_file']
  #det_out_param['save_output_param']['num_test_image'] = cfgs['net_cfgs']['num_test_image']
  det_eval_param['num_classes'] = det_out_param['num_classes']
  det_eval_param['background_label_id'] = det_out_param['background_label_id']
  #det_eval_param['name_size_file'] = det_out_param['save_output_param']['name_size_file']

def set_all_cfgs(cfgs):
  set_transform_cfgs(cfgs['transform_cfgs'])
  set_multibox_cfgs(cfgs['multibox_cfgs'])
  set_priorbox_cfgs(cfgs['prior_box_cfgs'])
  set_det_cfgs(cfgs)
  return batch_sampler, train_transform_param, test_transform_param, multibox_loss_param, loss_param, prior_box_param,solver_param, det_out_param, det_eval_param
