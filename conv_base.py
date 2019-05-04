import os, sys
import copy
sys.path.insert(0, '/ssd/caffe/python/')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

##pin = [from_layer,name, num_output,kernel_size, stride, group, bias]
def conv_params(*pin):
  ## must have from_layer, name, num_output, kenel_size
  #print len(pin)
  #print pin
  assert len(pin) >= 3
  pout = dict()
  from_layer = pin[0]
  name = pin[1]
  pout['num_output'] = pin[2]
  pout['bias_term'] = False
  pout['kernel_size'] = 1
  pout['param'] = [dict(lr_mult=1,decay_mult=1)]
  pout['weight_filler'] = dict(type='msra')
  if len(pin)>=4:
    pout['kernel_size'] = pin[3]
    pout['pad'] = int((pin[3]-1)/2)
  if len(pin)>=5:
    pout['stride'] = pin[4]
  if len(pin) >= 6:
    pout['group'] = pin[5]
    #name = "%s/dw"%name
    pout['engine'] = 1
  return from_layer, name, pout

##mb_params = [from_layer, name, cin, k_size, stride, c_out]
def mb_params(from_layer, name, cin, kernel_size, stride, cout):
  #print mb_para[0]
  _, _, dw_para = conv_params(from_layer, name, cin, kernel_size, stride, cin)
  #from_layer, name, dw_para = conv_params(*mb_para[0]) 
  _, _, linear_para = conv_params("", "", cout)
  expand = not (stride == 1)
  return from_layer, name, dw_para, linear_para, expand
  
def Conv(net, from_layer, name,  conv_para):
  net[name] = L.Convolution(net[from_layer], **conv_para) 
  return net

def ConvRelu(net, from_layer, name, conv_para):
  net[name] = L.Convolution(net[from_layer], **conv_para)
  relu_name = "%s/relu"%(name)
  net[relu_name] = L.ReLU(net[name], in_place=True)
  return net 
  
def ConvBNRelu(net, from_layer, name, conv_para, relu=True):
  bn_kwargs = {'param':[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0), dict(lr_mult=0,decay_mult=0)]}
  sc_kwargs = {'param':[dict(lr_mult=0,decay_mult=0), dict(lr_mult=0,decay_mult=0)],
               'bias_term':True,
               'filler': dict(type='constant', value=1.0),
               'bias_filler': dict(type='constant', value=0.0),}
  net[name] = L.Convolution(net[from_layer], **conv_para)
  bn_name = "%s/bn"%(name)
  net[bn_name] = L.BatchNorm(net[name], in_place=True, **bn_kwargs)
  sc_name = "%s/scale"%(name)
  net[sc_name] = L.Scale(net[name], in_place=True, **sc_kwargs)
  if relu:
    relu_name = "%s/relu"%(name)
    net[relu_name] = L.ReLU(net[name], in_place=True)
  return net 

def MobileBlock(net, from_layer, name, dw_para, linear_para, expand):
  dw_name = "%s/dw"%name
  net = ConvBNRelu(net, from_layer, dw_name, dw_para, relu=True)
  linear_name = "%s/linear"%name
  net = ConvBNRelu(net, dw_name, linear_name, linear_para, relu=False)
  if expand:
    expand_name = "%s/expand"%name
    expand_para = copy.deepcopy(linear_para)
    expand_para['stride'] = dw_para['stride']
    net = ConvBNRelu(net, from_layer, expand_name, expand_para, relu=False)
    from_layer = expand_name
  concat_name = "%s/concat"%name
  net[concat_name] = L.Eltwise(net[from_layer], net[linear_name])
  relu_name = "%s/relu"%(name)
  net[relu_name] = L.ReLU(net[concat_name], in_place=True)
  return net  

def construct_net_body(net, *np):
  for layer in np:
    #print layer
    l_type = layer.keys()[0]
    l_para = layer[l_type]
    if l_type == "conv":
      from_layer, name, conv_para = conv_params(*l_para)
      net = Conv(net, from_layer, name, conv_para)

    if l_type == "convR":
      from_layer, name, conv_para = conv_params(*l_para)
      net = ConvRelu(net, from_layer, name, conv_para)

    if l_type == "convBN":
      from_layer, name, conv_para = conv_params(*l_para)
      net = ConvBNRelu(net, from_layer, name, conv_para, relu=False)
      
    if l_type == "convBNR":
      from_layer, name, conv_para = conv_params(*l_para)
      net = ConvBNRelu(net, from_layer, name, conv_para, relu=True)

    if l_type == "mb":
      from_layer, name, dw_para, linear_para, expand = mb_params(*l_para) 
      net = MobileBlock(net, from_layer, name, dw_para, linear_para, expand)
  return net

if __name__ == '__main__':  
  def example_network():
    net = caffe.NetSpec()
  
    net.data = L.DummyData(shape=[dict(dim=[1]),
                                         dict(dim=[1])],
                                  transform_param=dict(scale=1.0/255.0),
                                  ntop=1)
    import yaml
    f = open('temp_net.yaml', 'r')
    net_paras = yaml.load(f)
    f.close()
    net = construct_net_body(net, *net_paras)
    '''
    cp0 = ['data', 'conv0', 32, 3]
    mbp = ['conv0', 'conv1', 32, 3, 2, 64]
    ##mb_params = [from_layer, name, cin, k_size, stride, c_out]
    #dwp = ['conv0', 'conv1', 32, 3, 2, 32]
    #lnp = ['', '', 64, 1]
    #mbp = [dwp, lnp, True]
  
    from_layer, name, c_para = conv_params(*cp0)
    n = Conv(n, from_layer, name, c_para)
    from_layer, name, dw_para, linear_para, expand = mb_params(*mbp)
    #mb_args  = mb_params(mbp)
    #n = MobileBlock(n, *mb_args)
    n = MobileBlock(n, from_layer, name, dw_para, linear_para, expand)
    #n  = ConvBNRelu(n, from_layer, name,  c_para, relu=False)
    '''
    return net.to_proto()
  
  temp = example_network()
  print temp
  
