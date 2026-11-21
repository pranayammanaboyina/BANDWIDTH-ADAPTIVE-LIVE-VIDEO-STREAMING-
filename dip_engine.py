import io
import cv2
import numpy as np
from PIL import Image

try:
    from dl_restorer import LightweightCNNRestorer
    _cnn_restorer = LightweightCNNRestorer()
except Exception:
    _cnn_restorer = None


BASE_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

BASE_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


class DIPProcessor:
    """
    Digital Image Processing + Deep Learning (AR-ResNet) Restoration Engine:
    - Spatial Pyramid Decimation (INTER_AREA)
    - ROI Motion-Aware Background Gaussian Filtering
    - Custom 8x8 DCT Quantization & Level-Shifted I/P Predictive Coding
    - Receiver Residual CNN (AR-ResNet) Deblocking + Unsharp Edge Restoration
    - 3-Frame Anti-Freeze Dense Optical Flow Motion Extrapolation
    - Full-Reference (MSE, PSNR, SSIM) & Spectral (Laplacian, 2D FFT) Metrics
    """

    _prev_motion_gray = None
    _conceal_streak = 0

    last_active_q = 72
    last_roi_ratio = 0.0
    last_edge_sharpness = 0.0
    last_fft_hf_ratio = 0.0

    @staticmethod
    def get_scaled_quant_matrix(quality: int):
        q = int(np.clip(quality, 1, 100))
        scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
        luma = np.clip(np.floor((BASE_LUMA_Q * scale + 50.0) / 100.0), 1, 255).astype(np.uint8)
        return luma.tolist()

    @staticmethod
    def _build_qtables(quality: int):
        q = int(np.clip(quality, 1, 100))
        scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
        luma = np.clip(np.floor((BASE_LUMA_Q * scale + 50.0) / 100.0), 1, 255).astype(np.uint8)
        chroma = np.clip(np.floor((BASE_CHROMA_Q * scale + 50.0) / 100.0), 1, 255).astype(np.uint8)
        return luma.flatten().tolist(), chroma.flatten().tolist()

    @classmethod
    def _apply_roi_filter(cls, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if cls._prev_motion_gray is None or cls._prev_motion_gray.shape != gray_blur.shape:
            cls._prev_motion_gray = gray_blur
            cls.last_roi_ratio = 1.0
            return frame_bgr

        diff = cv2.absdiff(gray_blur, cls._prev_motion_gray)
        cls._prev_motion_gray = gray_blur

        _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        roi_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        cls.last_roi_ratio = round(float(np.count_nonzero(roi_mask)) / float(roi_mask.size), 3)

        bg_smoothed = cv2.GaussianBlur(frame_bgr, (7, 7), 1.5)
        alpha = cv2.GaussianBlur(roi_mask.astype(np.float32) / 255.0, (11, 11), 0)[:, :, None]
        blended = frame_bgr.astype(np.float32) * alpha + bg_smoothed.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @classmethod
    def _encode_dct_jpeg(cls, frame_bgr: np.ndarray, quality: int) -> bytes:
        luma_list, chroma_list = cls._build_qtables(quality)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(
            buf,
            format="JPEG",
            qtables=[luma_list, chroma_list],
            subsampling="4:2:0",
            optimize=False
        )
        return buf.getvalue()

    @classmethod
    def process_frame(cls, frame_bgr, res_str, q_factor, prev_frame=None, force_keyframe=False):
        w, h = map(int, res_str.split("x"))
        downscaled = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        roi_filtered = cls._apply_roi_filter(downscaled)
        cls.last_active_q = int(q_factor)

        motion_high = False
        if prev_frame is not None and prev_frame.shape == roi_filtered.shape:
            mad = float(np.mean(np.abs(
                roi_filtered[::8, ::8, 0].astype(np.float32) - prev_frame[::8, ::8, 0].astype(np.float32)
            )))
            if mad > 28.0:
                motion_high = True

        use_i_frame = force_keyframe or motion_high or (prev_frame is None) or (prev_frame.shape != roi_filtered.shape)

        if use_i_frame:
            compressed_bytes = cls._encode_dct_jpeg(roi_filtered, q_factor)
            return compressed_bytes, downscaled, False
        else:
            residual = 0.5 * roi_filtered.astype(np.float32) - 0.5 * prev_frame.astype(np.float32) + 128.0
            residual_u8 = np.clip(residual, 0, 255).astype(np.uint8)
            compressed_bytes = cls._encode_dct_jpeg(residual_u8, q_factor)
            return compressed_bytes, downscaled, True

    @staticmethod
    def is_dl_active() -> bool:
        return (_cnn_restorer is not None) and _cnn_restorer.model_ready

    @staticmethod
    def post_process_frame(frame_bgr: np.ndarray, active_q: int = 50) -> np.ndarray:
        """
        Receiver DL Residual CNN (AR-ResNet) + Unsharp Mask Restoration for congested streams.
        """
        if active_q > 65:
            return frame_bgr

        if _cnn_restorer is not None and _cnn_restorer.model_ready:
            deblocked = _cnn_restorer.restore_frame(frame_bgr)
        else:
            deblocked = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=35, sigmaSpace=35)

        gaussian = cv2.GaussianBlur(deblocked, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(deblocked, 1.25, gaussian, -0.25, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @classmethod
    def decode_frame(cls, raw_bytes: bytes, is_p_frame: bool, prev_frame=None):
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            return None

        cls._conceal_streak = 0

        if not is_p_frame:
            if decoded.shape[1] <= 640:
                return cls.post_process_frame(decoded, active_q=45)
            return decoded

        if prev_frame is None or prev_frame.shape != decoded.shape:
            return None

        recon = prev_frame.astype(np.float32) + 2.0 * (decoded.astype(np.float32) - 128.0)
        recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
        if recon_u8.shape[1] <= 640:
            return cls.post_process_frame(recon_u8, active_q=45)
        return recon_u8

    @classmethod
    def synthesize_missing_frame(cls, prev2_bgr: np.ndarray, prev1_bgr: np.ndarray, prev3_bgr: np.ndarray = None):
        """
        Anti-Freeze Motion Synthesis Model:
        Extrapolates velocity + second-order acceleration across recent frames
        so video playback continues smoothly without freezing during packet loss.
        """
        if prev1_bgr is None:
            return None
        if prev2_bgr is None or prev2_bgr.shape != prev1_bgr.shape:
            return prev1_bgr

        h, w = prev1_bgr.shape[:2]
        sw, sh = max(32, w // 4), max(24, h // 4)

        g2 = cv2.cvtColor(cv2.resize(prev2_bgr, (sw, sh)), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cv2.resize(prev1_bgr, (sw, sh)), cv2.COLOR_BGR2GRAY)

        flow_v = cv2.calcOpticalFlowFarneback(g2, g1, None, 0.5, 2, 9, 2, 5, 1.1, 0)

        if prev3_bgr is not None and prev3_bgr.shape == prev1_bgr.shape:
            g3 = cv2.cvtColor(cv2.resize(prev3_bgr, (sw, sh)), cv2.COLOR_BGR2GRAY)
            flow_prev = cv2.calcOpticalFlowFarneback(g3, g2, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            accel = flow_v - flow_prev
            pred_flow = flow_v + 0.35 * accel
        else:
            pred_flow = flow_v

        cls._conceal_streak += 1
        damping = 0.85 ** cls._conceal_streak
        pred_flow = pred_flow * damping

        flow_full = cv2.resize(pred_flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_full[:, :, 0] *= (w / float(sw))
        flow_full[:, :, 1] *= (h / float(sh))

        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        synthesized = cv2.remap(
            prev1_bgr,
            gx - flow_full[:, :, 0],
            gy - flow_full[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return synthesized

    @classmethod
    def calculate_all_metrics(cls, ref_bgr: np.ndarray, rec_bgr: np.ndarray):
        ref_f = ref_bgr.astype(np.float32)
        rec_f = rec_bgr.astype(np.float32)
        mse = float(np.mean((ref_f - rec_f) ** 2))
        psnr = 99.0 if mse < 1e-6 else float(10.0 * np.log10((255.0 ** 2) / mse))

        ref_y = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)[::2, ::2].astype(np.float32)
        rec_y = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2GRAY)[::2, ::2].astype(np.float32)

        c1, c2 = 6.5025, 58.5225
        mu1 = cv2.GaussianBlur(ref_y, (7, 7), 1.5)
        mu2 = cv2.GaussianBlur(rec_y, (7, 7), 1.5)
        s1_sq = cv2.GaussianBlur(ref_y * ref_y, (7, 7), 1.5) - mu1 * mu1
        s2_sq = cv2.GaussianBlur(rec_y * rec_y, (7, 7), 1.5) - mu2 * mu2
        s12 = cv2.GaussianBlur(ref_y * rec_y, (7, 7), 1.5) - mu1 * mu2

        ssim_map = ((2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)) / (
            (mu1 ** 2 + mu2 ** 2 + c1) * (s1_sq + s2_sq + c2) + 1e-8
        )
        ssim = float(np.mean(ssim_map))

        cls.last_edge_sharpness = round(float(cv2.Laplacian(rec_y, cv2.CV_32F).var()), 1)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(rec_y)))
        fh, fw = mag.shape
        cy, cx = fh // 2, fw // 2
        ry, rx = max(1, fh // 6), max(1, fw // 6)
        tot_e = float(np.sum(mag)) + 1e-8
        low_e = float(np.sum(mag[cy - ry:cy + ry, cx - rx:cx + rx]))
        cls.last_fft_hf_ratio = round(max(0.0, (tot_e - low_e) / tot_e), 4)

        return mse, psnr, ssim

    @staticmethod
    def generate_diff_map(ref_bgr: np.ndarray, rec_bgr: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(ref_bgr, rec_bgr)
        amplified = np.clip(diff.astype(np.int32) * 5, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)