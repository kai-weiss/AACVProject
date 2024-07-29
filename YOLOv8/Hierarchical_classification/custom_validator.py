import warnings
from pathlib import Path

import torch
import os
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import callbacks


class CustomBaseValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, custom_predictions=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # --> MODI CODE START <-- #
        self.custom_predictions = custom_predictions
        # --> MODI CODE END <-- #


def use_custom_predictions(custom_predictions, current_file, predn_conf, predn_pred_cls):

    for pred in custom_predictions:

        if pred.path == current_file:

            for i in range(len(pred.boxes.cls)):

                if predn_pred_cls[i] != pred.boxes.cls[i]:
                    warnings.warn("DIFFERENT CLASSES")
                    # x = predn_pred_cls[i]
                    # y = pred.boxes.cls[i]

                predn_conf[i] = pred.boxes.conf[i]

            break

    return predn_conf


class CustomDetectionValidator(CustomBaseValidator, DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, custom_predictions=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks, custom_predictions)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def update_metrics(self, preds, batch):
        """Metrics."""
        # print("NEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEW")
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)

            # --> MODI CODE START <-- #
            current_file = batch["im_file"][si]

            # For hierarchical classification only the modification of the confidence score is needed.
            if self.custom_predictions:
                predn_pred_cls = predn[:, 5]
                predn_conf = use_custom_predictions(self.custom_predictions, current_file, predn[:, 4], predn_pred_cls)

                stat["conf"] = predn_conf
                stat["pred_cls"] = predn_pred_cls
            else:
                stat["conf"] = predn[:, 4]
                stat["pred_cls"] = predn[:, 5]
            # --> MODI CODE END <-- #

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)
