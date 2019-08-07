import os
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init
from collections import OrderedDict
from mmdet.models.utils import build_norm_layer

from mmdet.models.registry import BACKBONES

def common_conv2d(inplanes,
                  planes,
                  kernel,
                  padding,
                  stride,
                  norm_cfg=dict(type='BN')):
    """A common conv-bn-leakyrelu cell"""
    cell = OrderedDict()
    cell['conv'] = nn.Conv2d(inplanes, planes, kernel_size=kernel,
                             stride=stride, padding=padding, bias=False)
    if norm_cfg:
        norm_name, norm = build_norm_layer(norm_cfg, planes)
        cell[norm_name] = norm

    cell['leakyrelu'] = nn.LeakyReLU(0.1)
    cell = nn.Sequential(cell)
    return cell


class DarknetBasicBlockV3(nn.Module):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv."""

    def __init__(self,
                 inplanes,
                 planes,
                 norm_cfg=dict(type='BN')):
        super(DarknetBasicBlockV3, self).__init__()
        self.body = nn.Sequential(
            common_conv2d(inplanes, planes, 1, 0, 1, norm_cfg=norm_cfg),
            common_conv2d(planes, planes * 2, 3, 1, 1, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        residual = x
        x = self.body(x)
        return x + residual


@BACKBONES.register_module
class DarknetV3(nn.Module):

    def __init__(self,
                 layers=[1, 2, 8, 8, 4],
                 inplanes=[3, 32, 64, 128, 256, 512],
                 planes=[32, 64, 128, 256, 512, 1024],
                 num_stages=5,
                 classes=1000,
                 norm_cfg=dict(type='BN'),
                 out_indices=None,
                 frozen_stages=-1,
                 norm_eval=True):
        super(DarknetV3, self).__init__()
        assert len(layers) == len(planes) - 1, (
            "len(planes) should equal to len(layers) + 1, given {} vs {}".format(
                len(planes), len(layers)))
        self.num_stages = num_stages
        assert 5 >= num_stages >= 1
        assert max(out_indices) < num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.darknet_layer_name = []

        # first 3x3 conv
        self.stem = common_conv2d(inplanes[0], planes[0], 3, 1, 1, norm_cfg=norm_cfg)

        for i, (nlayer, inchannel, channel) in enumerate(zip(layers[:self.num_stages],
                                                             inplanes[1:self.num_stages + 1],
                                                             planes[1:self.num_stages + 1])):
            assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
            # add downsample conv with stride=2
            layer = []
            layer_name = 'layer{}'.format(i + 1)
            self.darknet_layer_name.append(layer_name)

            layer.append(common_conv2d(inchannel, channel, 3, 1, 2, norm_cfg=norm_cfg))

            # add nlayer basic blocks
            for _ in range(nlayer):
                layer.append(DarknetBasicBlockV3(channel, channel // 2))
            self.add_module(layer_name, nn.Sequential(*layer))

        if not self.out_indices:
            self.avg_gap = nn.AvgPool2d(7)
            self.output = nn.Linear(1024, classes)

        self._freeze_stages()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            import torch
            assert os.path.isfile(pretrained), "file {} not found.".format(pretrained)
            self.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        # D53: 18.1ms
        x = self.stem(x)
        outs = []
        for i, layer_name in enumerate(self.darknet_layer_name):
            darknet_layer = getattr(self, layer_name)
            x = darknet_layer(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)
        if not self.out_indices:
            x = self.avg_gap(x).view(x.size(0), -1)
            outs = self.output(x)
        return outs

    def train(self, mode=True):
        super(DarknetV3, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


if __name__ == '__main__':
    import numpy as np
    import torch
    from torchvision import transforms
    import cv2
    from PIL import Image
    import math
    from mmdet.core.utils.model_utils import describe_vars

    model = DarknetV3()
    describe_vars(model)

    '''The code below can be used to transfer the params from mxnet to pytorch.
    
    params = np.load('darknet53.npz')
    param_files = []
    for file in params.files:
        param_files.append(params[file])
    new_params = {}
    valid_i = 0
    for param_tensor in model.state_dict():
        if 'track' in param_tensor:
            continue
        new_params[param_tensor] = torch.tensor(param_files[valid_i], dtype=torch.float32)
        valid_i += 1
    model.load_state_dict(new_params)
    torch.save(model.state_dict(), 'darknet53.pth')
    '''

    model.init_weights('./pretrain/darknet53.pth')
    # model.load_state_dict(torch.load('darknet53.pk1'), strict=True)

    '''Print params in the pytorch model. 
    
    for param_tensor in model.state_dict():
        if param_tensor.startswith('stem.bn.'):
            print(param_tensor)
            print(model.state_dict()[param_tensor])
    '''

    img = cv2.imread('/private/liuzili/ILSVRC2012/val/ILSVRC2012_val_00028570.JPEG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resize = int(math.ceil(224 / 0.875))
    ratio = img.shape[0] / img.shape[1]
    new_size = (int(resize * ratio), resize) if ratio > 1 else (resize, int(resize / ratio))
    img = cv2.resize(img, (new_size[1], new_size[0]))

    def imshow(imgs, win_name='name_', wait_time=10000):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i, img in enumerate(imgs):
            cv2.imshow(win_name + str(i), img)
        cv2.waitKey(wait_time)

    nimg = Image.fromarray(np.array(img))

    nimg = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(nimg)

    for m in model.modules():
        m.eval()

    nimg = nimg.view(1, *nimg.size())
    out = model(nimg)
    print(torch.argmax(out))

    cv2.imwrite("tmp.jpg", img)
    # imshow(img)
