from PIL import Image
import torch
import numpy as np
import os
import runway

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from options.base_options import BaseOptions
from data.base_dataset import get_params, get_transform
import util.util as util

class Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')       
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(name='coco_pretrained')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        if not torch.cuda.is_available():
            parser.set_defaults(gpu_ids="-1")
        self.isTrain = False
        return parser

opt = Options().parse()

@runway.setup(options={'checkpoints_root': runway.file()})
def setup(opts):
    opt.checkpoints_dir = os.path.join(opts['checkpoints_root'], 'checkpoints')
    model = Pix2PixModel(opt)
    model.eval()
    return model

@runway.command('convert', inputs={'semantic_map': runway.image(channels=1), 'reference': runway.image}, outputs={'output': runway.image})
def convert(model, inputs):
    img = inputs['semantic_map']
    reference = inputs['reference'].convert('RGB')
    params = get_params(opt, img.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(img) * 255.0
    label_tensor[label_tensor == 255.0] = opt.label_nc
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(reference)
    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    output = util.tensor2im(generated[0])
    output = Image.fromarray(output)
    return output

if __name__ == '__main__':
    runway.run(model_options={'checkpoints_root': '.'})
