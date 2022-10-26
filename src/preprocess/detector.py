import cv2
import numpy as np
import omegaconf
import torch
import torchvision


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def letterbox(
    img: np.array,
    new_shape: tuple,
    color: tuple = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> np.array:
    """
    Perform letterbox padding.

    Parameters
    ----------
    img : np.array
        Input image.

    new_shape : tuple
        New shape of the padded image.

    color : tuple
        Color of the border.

    auto : bool
        True if minimum rectangle mode.

    scale_fill : bool
        True if padded to new_shape.

    scaleup : bool
        Scale up (?)

    stride : int
        Stride of padding

    Returns
    -------
        Padded image (np.array).
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


class BarcodeDetector(object):
    """Barcode detector."""

    def __init__(self, config: dict | omegaconf.dictconfig.DictConfig):
        """
        Initialize Barcode Detector.

        Parameters
        ----------
        config : dict
            Configuration dictionary with the following structure:
                model_path : path to TorchScript model
                device : device
                img_size : required input image size
                conf_thresh : confidence threshold
                iou_thresh : IoU threshold
        """
        self._model = torch.jit.load(config['model_path'], map_location=config['device'])
        self._model.eval()
        self._device = config['device']
        self._img_size = config['img_size']
        self._conf_thresh = config['conf_thresh']
        self._iou_thresh = config['iou_thresh']

    def detect(self, image: np.array):
        """
        Detect barcode on the photo and return cropped by its bounding box image.

        Parameters
        ----------
        image : np.array
            Input image

        Returns
        -------
            Cropped image of the barcode.
        """
        input_image = self._preprocess_image(image)
        output = self._model(input_image)
        prediction = self._non_max_suppression(output)
        bbox = self._postprocess(prediction, orig_image_shape=image.shape, input_image_shape=input_image.shape)
        return self._crop_by_bbox(image, bbox)

    def _preprocess_image(self, image: np.array) -> torch.tensor:
        """
        Preprocess image for YoloV5 model.

        Parameters
        ----------
        image : np.array
            Input image

        Returns
        -------
            Preprocessed image (torch.tensor)
        """
        max_uint8 = 255
        new_shape = (self._img_size, self._img_size)
        image = letterbox(image, new_shape=new_shape, auto=False)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.float() / max_uint8
        if image.ndim == 3:
            image = image[None]
        return image.to(self._device)

    def _non_max_suppression(  # noqa: WPS231
        self,
        output: torch.tensor,
        agnostic: bool = False,
        max_det: int = 300,
        nm: int = 0,  # number of masks
    ) -> torch.tensor:
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

        Parameters
        ----------
        output : torch.tensor
            Raw output of the detector.

        agnostic : bool
            Agnostic NMS

        max_det : int
            Maximum detections number.

        nm : int
            Number of masks.

        Returns
        -------
            List of detections, on (n,6) tensor per image [xyxy, conf, cls].
        """
        if isinstance(output, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = output[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > self._conf_thresh  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device) for _ in range(bs)]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > self._conf_thresh]  # noqa: WPS440

            # Check shape
            n = x.shape[0]  # number of boxes
            if n:  # no boxes
                if n > max_nms:  # excess boxes
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
                else:
                    x = x[x[:, 4].argsort(descending=True)]  # sort by confidence
            else:
                continue
            # Batched NMS
            classes_coeff = 0 if agnostic else max_wh
            c = x[:, 5:6] * classes_coeff  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self._iou_thresh)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
        return output[0].cpu()

    def _postprocess(self, prediction: torch.tensor, orig_image_shape: tuple, input_image_shape: tuple):
        prediction[:, :4] = scale_boxes(input_image_shape[2:], prediction[:, :4], orig_image_shape).round()
        bbox = prediction[0][:4].long()
        return {
            'x1': bbox[0],
            'y1': bbox[1],
            'x2': bbox[2],
            'y2': bbox[3],
        }

    def _crop_by_bbox(self, image: np.array, bbox: dict):
        y1, y2, x1, x2 = bbox['y1'], bbox['y2'], bbox['x1'], bbox['x2']
        if image.ndim == 3:
            return image[y1:y2, x1:x2, :]
        return image[:, y1:y2, x1:x2, :]
