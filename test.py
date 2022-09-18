class MiniDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        yaml: str,
        confidence: float,
    ):
        self.model_path = model_path
        self.device = device
        self.yaml = yaml
        self.confidence = confidence
        self.load()
    
    def load(self):
        from module.detector import Detector
        from utils.tool import LoadYaml
        import torch

        cfg = LoadYaml(self.yaml)
        model = Detector(cfg.category_num, True).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval() 
         
        self.model = model
        self.cfg = cfg
    
    def object_prediction_list(self, img):
        from utils.tool import handle_preds
        from utils.visualize import vis
        import torch
        import cv2
        import time

        ori_img = cv2.imread(img)
        res_img = cv2.resize(ori_img, (self.cfg.input_width, self.cfg.input_height), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg.input_height, self.cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0

        start = time.time()
        preds = self.model(img)
        end = time.time()
        time = (end - start)
        print(f"Time: {time:.2f} s, FPS: {1 / time:.2f}")
        output = handle_preds(preds, self.device, self.confidence)

        LABEL_NAMES = []
        with open(self.cfg.names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
                
        H, W, _ = ori_img.shape
        prediction_list = []
        for box in output[0]:
            box = box.tolist()
            score = box[4]
            category = LABEL_NAMES[int(box[5])]
            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            bbox = [x1, y1, x2, y2]
            prediction_list.append(
                {
                    "bbox": bbox, 
                    "score": score, 
                    "category_name": category, 
                    "category_id": box[5]
                }
            )
        vis(ori_img, prediction_list)

MiniDetector(
    model_path= 'data/weights/50-epoch.pth',
    device= 'cpu',
    yaml= 'drone_data/drone.yaml',
    confidence= 0.5,
).object_prediction_list('drone_data/val/0001.jpg')