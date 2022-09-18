import torch
import argparse
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector

device = torch.device("cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="data/coco.yaml", help='.yaml config')
    parser.add_argument('--weight', type=str, default="data/weights/model.pth", help='.weight config')
    
    opt = parser.parse_args()
    cfg = LoadYaml(opt.yaml)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight))
    model.eval()

    #summary(model, input_size=(3, cfg.input_height, cfg.input_width))
    evaluation = CocoDetectionEvaluator(cfg.names, device)
    val_dataset = TensorDataset(cfg.val_txt, cfg.input_width, cfg.input_height, False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    print("computer mAP...")
    evaluation.compute_map(val_dataloader, model)

