import albumentations as alb
import numpy as np
import onnxruntime


class OCRecogniser(object):
    """Optical Character Recogniser class."""

    def __init__(self, config: dict):
        """
        Initialize OCRecogniser.

        Parameters
        ----------
        config : dict
            Configuration dictionary with the following structure:
                model_path : str -- path to ONNX model.
                blank : int -- blank symbol code.
                img_height : int -- image height
                img_width : int -- image width
        """
        self._ort_session = onnxruntime.InferenceSession(config['model_path'])
        self._blank = config['blank']
        self._transform = alb.Compose(
            [
                alb.SmallestMaxSize(
                    max_size=config['img_height'],
                    interpolation=0,
                    always_apply=True,
                ),
                alb.PadIfNeeded(
                    min_height=config['img_height'],
                    min_width=config['img_width'],
                    border_mode=0,
                    value=(0, 0, 0),
                    always_apply=True,
                ),
            ],
        )

    def predict(self, images: np.array) -> list[str]:
        """
        Predict on batch of the images.

        Parameters
        ----------
        images : np.array
            Batch of images. Shape: [bs, 3, img_height, img_width]

        Returns
        -------
        list[str]
            List with recognised texts.

        """
        pred_texts = []
        for image in images:
            image = self._preprocess(image)
            pred = self._predict(image)
            pred_text = self._pred2text(pred[0])
            pred_texts.append(pred_text)
        return pred_texts

    def _preprocess(self, image: np.array) -> np.array:
        maxuint8 = 255
        image = self._transform(image=image)
        image = image['image']
        image = (image / maxuint8).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return image[None, ...]

    def _predict(self, image: np.array) -> np.array:
        ort_inputs = {self._ort_session.get_inputs()[0].name: image}
        preds = self._ort_session.run(None, ort_inputs)
        pred_texts = np.argmax(preds[0], -1)
        return np.transpose(pred_texts, (1, 0))

    def _pred2text(self, pred: np.array) -> str:
        """
        Prediction to text.

        E.g. AA---A-BBB--C-C -> AABCC.

        Parameters
        ----------
        pred : np.array
            Prediction of the OCR model (probabilities of each symbol). Shape: [time_steps, 1]

        Returns
        -------
        str
            Recognised text
        """
        text = ''

        if pred[0] != self._blank:
            text += str(pred[0])

        ind = 1
        while ind < len(pred):
            if pred[ind - 1] == pred[ind]:
                ind += 1
                continue
            elif pred[ind] != self._blank:
                text += str(pred[ind])
            ind += 1
        return text
