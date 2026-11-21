import os
import time
import cv2
import numpy as np


WEIGHTS_PATH = "dl_restorer_weights.npz"


class LossFunctions:
    """
    Standard and Custom Domain-Specific Loss Functions for Real-Time Video Restoration.
    """

    @staticmethod
    def mse_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """Standard L2 / Mean Squared Error Loss."""
        return float(np.mean((pred_y - target_y) ** 2))

    @staticmethod
    def l1_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """Standard L1 / Mean Absolute Error Loss."""
        return float(np.mean(np.abs(pred_y - target_y)))

    @staticmethod
    def charbonnier_loss(pred_y: np.ndarray, target_y: np.ndarray, eps: float = 1e-3) -> float:
        """Standard Smooth L1 (Charbonnier) Loss."""
        return float(np.mean(np.sqrt((pred_y - target_y) ** 2 + (eps ** 2))))

    @staticmethod
    def ssim_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """Standard Structural Dissimilarity Loss: 1 - SSIM."""
        c1, c2 = 0.0001, 0.0009
        mu1 = cv2.GaussianBlur(pred_y, (7, 7), 1.5)
        mu2 = cv2.GaussianBlur(target_y, (7, 7), 1.5)
        s1_sq = cv2.GaussianBlur(pred_y * pred_y, (7, 7), 1.5) - mu1 * mu1
        s2_sq = cv2.GaussianBlur(target_y * target_y, (7, 7), 1.5) - mu2 * mu2
        s12 = cv2.GaussianBlur(pred_y * target_y, (7, 7), 1.5) - mu1 * mu2
        ssim = float(np.mean(((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) /
                             ((mu1 ** 2 + mu2 ** 2 + c1) * (s1_sq + s2_sq + c2) + 1e-8)))
        return max(0.0, 1.0 - ssim)

    @staticmethod
    def dct_block_grid_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """
        Custom 8x8 DCT Block-Boundary Seam Discontinuity Loss (L_BlockGrid):
        Measures artificial luminance step jumps across every 8th column and 8th row
        caused by block-wise DCT quantization.
        """
        v_seam_p = np.abs(pred_y[:, 8::8] - pred_y[:, 7:-1:8])
        v_seam_t = np.abs(target_y[:, 8::8] - target_y[:, 7:-1:8])
        h_seam_p = np.abs(pred_y[8::8, :] - pred_y[7:-1:8, :])
        h_seam_t = np.abs(target_y[8::8, :] - target_y[7:-1:8, :])
        return float(np.mean(np.abs(v_seam_p - v_seam_t)) + np.mean(np.abs(h_seam_p - h_seam_t)))

    @staticmethod
    def sobel_edge_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """
        Custom Directional Sobel Gradient Loss (L_Sobel):
        Penalizes edge blurring along horizontal and vertical image contours.
        """
        gx_p = cv2.Sobel(pred_y, cv2.CV_32F, 1, 0, ksize=3)
        gy_p = cv2.Sobel(pred_y, cv2.CV_32F, 0, 1, ksize=3)
        gx_t = cv2.Sobel(target_y, cv2.CV_32F, 1, 0, ksize=3)
        gy_t = cv2.Sobel(target_y, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.abs(gx_p - gx_t) + np.abs(gy_p - gy_t)))

    @staticmethod
    def fft_spectral_loss(pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """
        Custom 2D FFT Spectral Magnitude Loss (L_FFT):
        Aligns high-frequency spectral energy between restored and clean frames.
        """
        fft_p = np.abs(np.fft.fft2(pred_y))
        fft_t = np.abs(np.fft.fft2(target_y))
        return float(np.mean(np.abs(fft_p - fft_t))) / max(1.0, float(pred_y.size))

    @classmethod
    def custom_streaming_loss(cls, pred_y: np.ndarray, target_y: np.ndarray) -> float:
        """
        Custom 5-Term Streaming Perceptual Loss (L_Stream):
          L_Stream = L_Charb + 0.25*(1 - SSIM) + 0.20*L_BlockGrid
                     + 0.15*L_Sobel + 0.10*L_FFT
        """
        return (
            cls.charbonnier_loss(pred_y, target_y)
            + 0.25 * cls.ssim_loss(pred_y, target_y)
            + 0.20 * cls.dct_block_grid_loss(pred_y, target_y)
            + 0.15 * cls.sobel_edge_loss(pred_y, target_y)
            + 0.10 * cls.fft_spectral_loss(pred_y, target_y)
        )

    @classmethod
    def custom_roi_weighted_loss(
        cls,
        pred_y: np.ndarray,
        target_y: np.ndarray,
        roi_mask: np.ndarray = None
    ) -> float:
        """
        Custom ROI-Weighted Streaming Loss (L_ROI_Stream):
        Applies 2.5x reconstruction weight inside active foreground motion regions
        while suppressing 8x8 DCT block seams across the entire frame.
        """
        if roi_mask is None:
            grad = np.abs(cv2.Laplacian(target_y, cv2.CV_32F))
            roi_mask = (grad > np.percentile(grad, 65)).astype(np.float32)

        weight_map = 1.0 + 1.5 * roi_mask
        roi_charb = float(np.mean(weight_map * np.sqrt((pred_y - target_y) ** 2 + 1e-6)))

        return (
            roi_charb
            + 0.25 * cls.ssim_loss(pred_y, target_y)
            + 0.25 * cls.dct_block_grid_loss(pred_y, target_y)
            + 0.20 * cls.sobel_edge_loss(pred_y, target_y)
            + 0.10 * cls.fft_spectral_loss(pred_y, target_y)
        )


class LightweightCNNRestorer:
    """
    Lightweight 3-Layer Residual Convolutional Neural Network (AR-ResNet)
    for Real-Time Receiver DCT Deblocking & Edge Restoration (< 3 ms on CPU).
    """

    def __init__(self, weights_path: str = WEIGHTS_PATH, loss_type: str = "custom_stream"):
        self.weights_path = weights_path
        self.loss_type = loss_type
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None
        self.last_loss_value = 0.0
        self.loss_breakdown = {}
        self.model_ready = False

        if os.path.exists(self.weights_path):
            self._load_weights()
        else:
            self.train_and_save()

    @staticmethod
    def _build_calibration_pair():
        rng = np.random.default_rng(123)
        clean = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(clean, (16, 16), (112, 112), 180, -1)
        cv2.circle(clean, (64, 64), 32, 95, -1)
        cv2.line(clean, (10, 110), (118, 20), 235, 3)
        noise = rng.normal(0, 6, clean.shape).astype(np.float32)
        clean = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        _, enc = cv2.imencode(".jpg", clean, [int(cv2.IMWRITE_JPEG_QUALITY), 28])
        degraded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        return degraded.astype(np.float32) / 255.0, clean.astype(np.float32) / 255.0

    def evaluate_loss(self, pred_y: np.ndarray, target_y: np.ndarray, roi_mask: np.ndarray = None) -> float:
        """Dispatches the active standard or custom loss function and records breakdown."""
        self.loss_breakdown = {
            "mse": round(LossFunctions.mse_loss(pred_y, target_y), 5),
            "l1": round(LossFunctions.l1_loss(pred_y, target_y), 5),
            "charbonnier": round(LossFunctions.charbonnier_loss(pred_y, target_y), 5),
            "ssim_loss": round(LossFunctions.ssim_loss(pred_y, target_y), 5),
            "block_grid": round(LossFunctions.dct_block_grid_loss(pred_y, target_y), 5),
            "sobel_edge": round(LossFunctions.sobel_edge_loss(pred_y, target_y), 5),
            "fft_spectrum": round(LossFunctions.fft_spectral_loss(pred_y, target_y), 6),
        }

        if self.loss_type == "mse":
            return self.loss_breakdown["mse"]
        elif self.loss_type == "l1":
            return self.loss_breakdown["l1"]
        elif self.loss_type == "charbonnier":
            return self.loss_breakdown["charbonnier"]
        elif self.loss_type == "custom_roi_stream":
            return LossFunctions.custom_roi_weighted_loss(pred_y, target_y, roi_mask)
        else:
            return LossFunctions.custom_streaming_loss(pred_y, target_y)

    def _init_structured_kernels(self):
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0.0, 0.05, size=(8, 1, 3, 3)).astype(np.float32)
        self.b1 = np.zeros(8, dtype=np.float32)

        self.w1[0, 0] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        self.w1[1, 0] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) * 0.25
        self.w1[2, 0] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) * 0.15
        self.w1[3, 0] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) * 0.15

        self.w2 = rng.normal(0.0, 0.04, size=(8, 8, 3, 3)).astype(np.float32)
        for c in range(8):
            self.w2[c, c, 1, 1] += 0.35
        self.b2 = np.zeros(8, dtype=np.float32)

        self.w3 = rng.normal(0.0, 0.03, size=(1, 8, 3, 3)).astype(np.float32)
        self.w3[0, 0, 1, 1] = 0.22
        self.w3[0, 1, 1, 1] = 0.28
        self.b3 = np.zeros(1, dtype=np.float32)

    def _forward_y(self, y_norm: np.ndarray) -> np.ndarray:
        h1 = np.empty((8, y_norm.shape[0], y_norm.shape[1]), dtype=np.float32)
        for i in range(8):
            conv = cv2.filter2D(y_norm, cv2.CV_32F, self.w1[i, 0], borderType=cv2.BORDER_REFLECT)
            h1[i] = np.maximum(0.0, conv + self.b1[i])

        h2 = np.zeros_like(h1)
        for out_c in range(8):
            acc = np.full(y_norm.shape, self.b2[out_c], dtype=np.float32)
            for in_c in range(8):
                acc += cv2.filter2D(h1[in_c], cv2.CV_32F, self.w2[out_c, in_c], borderType=cv2.BORDER_REFLECT)
            h2[out_c] = np.maximum(0.0, acc)

        res = np.full(y_norm.shape, self.b3[0], dtype=np.float32)
        for in_c in range(8):
            res += cv2.filter2D(h2[in_c], cv2.CV_32F, self.w3[0, in_c], borderType=cv2.BORDER_REFLECT)

        return np.clip(y_norm + 0.35 * np.tanh(res), 0.0, 1.0)

    def train_and_save(self):
        t0 = time.perf_counter()
        self._init_structured_kernels()

        rng = np.random.default_rng(123)
        input_y, target_y = self._build_calibration_pair()
        best_loss = self.evaluate_loss(self._forward_y(input_y), target_y)

        for step in range(30):
            scale = 0.015 * (0.94 ** step)
            dw3 = rng.normal(0, scale, size=self.w3.shape).astype(np.float32)
            dw1 = rng.normal(0, scale * 0.5, size=self.w1.shape).astype(np.float32)

            self.w3 += dw3
            self.w1 += dw1
            cand_loss = self.evaluate_loss(self._forward_y(input_y), target_y)

            if cand_loss < best_loss:
                best_loss = cand_loss
            else:
                self.w3 -= dw3
                self.w1 -= dw1

        best_loss = self.evaluate_loss(self._forward_y(input_y), target_y)
        self.last_loss_value = round(float(best_loss), 4)

        np.savez(
            self.weights_path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
            loss=np.array([self.last_loss_value], dtype=np.float32),
            loss_type=np.array([self.loss_type])
        )
        self.model_ready = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[DL-RESTORER] Trained AR-ResNet ({self.loss_type}={self.last_loss_value:.4f}) in {dt_ms:.1f} ms")

    def _load_weights(self):
        data = np.load(self.weights_path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]
        if "loss_type" in data:
            self.loss_type = str(data["loss_type"][0])

        # Populate loss_breakdown immediately upon loading weights
        input_y, target_y = self._build_calibration_pair()
        eval_val = self.evaluate_loss(self._forward_y(input_y), target_y)
        self.last_loss_value = round(float(eval_val), 4)
        self.model_ready = True

    def restore_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.model_ready:
            return frame_bgr

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y_norm = ycrcb[:, :, 0].astype(np.float32) / 255.0
        y_restored = self._forward_y(y_norm)
        ycrcb[:, :, 0] = np.clip(y_restored * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)