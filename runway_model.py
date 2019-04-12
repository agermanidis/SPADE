from PIL import Image
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
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(name='coco_pretrained')
        self.isTrain = False
        return parser

opt = Options().parse()

@runway.setup(options={'checkpoints_root': runway.file()})
def setup(opts):
    opt.checkpoints_dir = os.path.join(opts['checkpoints_root'], 'checkpoints')
    if not torch.cuda.is_available():
        opt.gpu_ids = -1
    model = Pix2PixModel(opt)
    model.eval()
    return model

@runway.command('convert', inputs={'semantic_map': runway.image(channels=1), 'reference': runway.image}, outputs={'output': runway.image})
def convert(model, inputs):
    img = np.array(inputs['semantic_map'])
    reference = inputs['reference']
    h, w = img.shape[0:2]
    img = Image.fromarray(img)
    params = get_params(opt, (w, h))
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(img).unsqueeze(0)
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(reference)
    data = {
        'label': label_tensor,
        'instance': label_tensor,
        'image': image_tensor
    }
    generated = model(data, mode='inference')
    output = util.tensor2im(generated[0])
    output = Image.fromarray(output)
    return output

if __name__ == '__main__':
    runway.run(model_options={'checkpoints_root': '.'})
