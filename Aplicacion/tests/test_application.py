"""Comprobaciones de integración del corte DICOM y los artefactos locales."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_model_configs
from core.inference import predict
from core.model_loader import load_model
from core.preprocessing import prepare_dicom, preprocess


def dicom_bytes() -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1000
    pixels = np.arange(256, dtype=np.int16).reshape(16, 16) * 12
    dataset.PixelData = pixels.tobytes()
    stream = BytesIO()
    pydicom.dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


class ApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = prepare_dicom(dicom_bytes(), 40_000_000)

    def test_dicom_window_and_channels(self) -> None:
        models, errors = load_model_configs()
        self.assertEqual(errors, [])
        self.assertEqual(self.image.size, (224, 224))
        array = np.asarray(self.image)
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(int(array[0, 0]), 0)
        self.assertEqual(int(array[-1, -1]), 255)
        for config in models:
            batch = preprocess(self.image, config)
            self.assertEqual(batch.shape, (1, 224, 224, config.channels))
            if config.channels == 3:
                np.testing.assert_array_equal(batch[0, :, :, 0], batch[0, :, :, 1])

    def test_trained_models_accept_dicom_input(self) -> None:
        models, errors = load_model_configs()
        self.assertEqual(errors, [])
        if not all(config.path.is_file() for config in models):
            self.skipTest("Los artefactos .keras no están presentes")
        for config in models:
            with self.subTest(model=config.name):
                runtime = load_model(config.cache_key, config.as_dict())
                result = predict(runtime, config, self.image)
                self.assertTrue(0 <= result.fracture_score <= 1)
                self.assertIn(result.predicted_class, config.class_order)


if __name__ == "__main__":
    unittest.main()
