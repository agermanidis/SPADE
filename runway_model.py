from PIL import Image
import torch
import numpy as np
import os
import runway
import argparse

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from options.base_options import BaseOptions
from data.base_dataset import get_params, get_transform
import util.util as util
from util.coco import label_to_id

class Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')       
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        parser.set_defaults(load_from_opt_file=True)
        if not torch.cuda.is_available():
            parser.set_defaults(gpu_ids="-1")
        self.isTrain = False
        return parser


@runway.setup(options={'checkpoints_root': runway.file(is_directory=True)})
def setup(opts):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options = Options()
    parser = options.initialize(parser)
    options.parser = parser
    parser.set_defaults(name=opts['checkpoints_root'].split('/')[-1])
    parser.set_defaults(checkpoints_dir=os.path.join(opts['checkpoints_root'], '..'))
    model = Pix2PixModel(options.parse())
    model.eval()
    return model

command_inputs = {
    'semantic_map': runway.segmentation(label_to_id=label_to_id, width=512, height=512),
}

command_outputs = {
    'output': runway.image
}

@runway.command('convert', inputs=command_inputs, outputs=command_outputs)
def convert(model, inputs):
    img = inputs['semantic_map']
    original_size = img.size
    img = img.resize((512, 512))
    params = get_params(opt, img.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(img).unsqueeze(0)
    data = {
        'label': label_tensor,
        'instance': label_tensor,
        'image': None
    }
    generated = model(data, mode='inference')
    output = util.tensor2im(generated[0])
    output = Image.fromarray(output).resize(original_size)
    return output

if __name__ == '__main__':
    runway.run(port=5132, debug=True)
