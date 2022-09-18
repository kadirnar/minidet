import torch
import argparse
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="data/coco.yaml", help='.yaml config')
    parser.add_argument('--weight', type=str, default="data/weights/model.pth", help='.weight config')
    parser.add_argument('--device', type=str, default="cuda:0", help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=list, default=[416, 416], help='size of each image dimension')
    opt = parser.parse_args()
    cfg = LoadYaml(opt.yaml)
    model = Detector(cfg.category_num, True).to(opt.device)
    model.load_state_dict(torch.load(opt.weight))
    model.eval()

    #summary(model, input_size=(3, cfg.input_height, cfg.input_width))
    evaluation = CocoDetectionEvaluator(cfg.names, opt.device)
    val_dataset = TensorDataset(cfg.val_txt, opt.img_size[0], opt.img_size[1], False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    print("computer mAP...")
    evaluation.compute_map(val_dataloader, model)

