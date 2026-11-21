import time
import numpy as np
from collections import deque


class AdaptiveBitrateController:
    """
    Computer Networks (CN) Rule-Based Adaptive Bitrate (ABR) Controller
    with Asymmetric AIMD Hysteresis and Online Empirical Quality Telemetry.

    Defaults to standard 720p HD (P4) when link capacity is healthy (6000 kbps)
    and dynamically steps down to P3/P2/P1/P0 under bandwidth caps, packet loss,
    RTT spikes, or jitter.
    """

    PROFILES = [
        {
            "name": "P0 (240p Low-BW)",
            "res": "426x240",
            "fps": 30,
            "q": 35,
            "q_min": 20,
            "q_max": 42,
            "target_kbps": 350,
            "prior_ssim": 0.76,
            "prior_psnr": 28.5
        },
        {
            "name": "P1 (360p Guard)",
            "res": "640x360",
            "fps": 30,
            "q": 50,
            "q_min": 35,
            "q_max": 60,
            "target_kbps": 1000,
            "prior_ssim": 0.84,
            "prior_psnr": 32.0
        },
        {
            "name": "P2 (480p Balanced)",
            "res": "854x480",
            "fps": 30,
            "q": 72,
            "q_min": 55,
            "q_max": 78,
            "target_kbps": 2200,
            "prior_ssim": 0.90,
            "prior_psnr": 35.8
        },
        {
            "name": "P3 (480p Crisp)",
            "res": "854x480",
            "fps": 30,
            "q": 85,
            "q_min": 72,
            "q_max": 90,
            "target_kbps": 3600,
            "prior_ssim": 0.94,
            "prior_psnr": 39.2
        },
        {
            "name": "P4 (720p HD)",
            "res": "1280x720",
            "fps": 30,
            "q": 95,
            "q_min": 82,
            "q_max": 95,
            "target_kbps": 6000,
            "prior_ssim": 0.98,
            "prior_psnr": 42.5
        },
    ]

    def __init__(self, validation_gate_samples: int = 100):
        self.current_idx = 4  # Standard 720p HD (P4) baseline
        self.last_switch_time = time.time()
        self.last_switch_reason = "Initialized at P4 (720p HD Standard)"
        self.upward_streak = 0
        self.validation_gate_samples = validation_gate_samples

        # Live Telemetry State
        self.last_safe_bw_kbps = 6000.0
        self.last_net_stress = 0.0
        self.all_profile_status = {}
        self.all_profile_congestion = {}
        self.all_profile_est_psnr = {}

        # Empirical Quality Calibration (Online Bias Offset)
        self.psnr_bias_per_tier = {i: 0.0 for i in range(len(self.PROFILES))}
        self.pred_psnr_errors = deque(maxlen=200)
        self.congestion_hits = deque(maxlen=200)
        self.history_records = deque(maxlen=600)

        self._bootstrap_rd_curve()

    def _bootstrap_rd_curve(self):
        rng = np.random.default_rng(42)
        for _ in range(320):
            p = int(rng.integers(0, len(self.PROFILES)))
            prof = self.PROFILES[p]
            avail_kbps = float(rng.uniform(250, 7500))
            loss_pct = float(rng.exponential(3.5))
            rtt_ms = float(rng.uniform(5, 180))
            jitter_ms = float(rng.uniform(1, 35))

            est_psnr = prof["prior_psnr"] - 0.14 * min(loss_pct, 25.0)
            self.history_records.append({
                "profile": p,
                "avail_kbps": avail_kbps,
                "loss_pct": loss_pct,
                "rtt_ms": rtt_ms,
                "jitter_ms": jitter_ms,
                "est_psnr": round(float(np.clip(est_psnr, 22.0, 46.0)), 2)
            })

    def get_current_profile(self):
        return self.PROFILES[self.current_idx]

    def record_outcome(
        self,
        avail_kbps: float,
        loss_pct: float,
        rtt_ms: float,
        jitter_ms: float,
        profile_idx: int,
        actual_psnr: float,
        concealed_or_dropped: bool
    ):
        prof = self.PROFILES[profile_idx]
        est_psnr = prof["prior_psnr"] + self.psnr_bias_per_tier[profile_idx] - 0.14 * min(loss_pct, 25.0)
        err = actual_psnr - est_psnr

        self.psnr_bias_per_tier[profile_idx] = float(
            np.clip(self.psnr_bias_per_tier[profile_idx] + 0.08 * err, -6.0, 6.0)
        )

        predicted_congested = (prof["target_kbps"] > self.last_safe_bw_kbps) or (self.last_net_stress > 0.45)
        actual_congested = bool(concealed_or_dropped or loss_pct > 2.5)

        self.pred_psnr_errors.append(err)
        self.congestion_hits.append(1.0 if predicted_congested == actual_congested else 0.0)
        self.history_records.append({
            "profile": profile_idx,
            "avail_kbps": avail_kbps,
            "loss_pct": loss_pct,
            "rtt_ms": rtt_ms,
            "jitter_ms": jitter_ms,
            "est_psnr": round(actual_psnr, 2)
        })

    def get_validation_stats(self) -> dict:
        n = len(self.pred_psnr_errors)
        if n == 0:
            return {
                "live_samples": 0,
                "gate_passed": False,
                "psnr_mae": 0.0,
                "psnr_rmse": 0.0,
                "hit_rate_pct": 100.0,
            }

        errs = np.array(self.pred_psnr_errors, dtype=np.float32)
        mae = float(np.mean(np.abs(errs)))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        hit_rate = float(np.mean(self.congestion_hits) * 100.0)

        return {
            "live_samples": n,
            "gate_passed": n >= self.validation_gate_samples,
            "psnr_mae": round(mae, 2),
            "psnr_rmse": round(rmse, 2),
            "hit_rate_pct": round(hit_rate, 1),
        }

    def evaluate(
        self,
        measured_loss: float,
        measured_rtt: int,
        measured_jitter: float,
        configured_bw_kbps: float,
        achieved_kbps: float
    ):
        """
        Evaluates safe link bandwidth and congestion thresholds across P0..P4.
        Maintains P4 (720p HD) when link capacity is >= 5500 kbps and loss/RTT are low;
        steps down immediately on congestion and probes upward smoothly.
        """
        now = time.time()
        loss_pct = measured_loss * 100.0

        # 1. Estimate safe deliverable link bandwidth envelope B_safe from shaper/link capacity
        loss_discount = max(0.20, 1.0 - 2.5 * measured_loss)
        safe_bw = float(configured_bw_kbps) * loss_discount

        if measured_rtt > 130 or measured_jitter > 20.0:
            safe_bw *= 0.65
        elif measured_rtt > 75 or measured_jitter > 10.0:
            safe_bw *= 0.82

        self.last_safe_bw_kbps = round(safe_bw, 1)

        # 2. Compute normalized deterministic network congestion stress [0.0, 1.0]
        net_stress = min(
            1.0,
            (measured_loss / 0.12) * 0.55
            + (max(0, measured_rtt - 20) / 140.0) * 0.25
            + (max(0.0, measured_jitter - 2.0) / 25.0) * 0.20
        )
        self.last_net_stress = round(net_stress, 4)

        # 3. Evaluate bandwidth fit & congestion caps across all 5 profiles P0..P4
        status_map = {}
        tier_congestion_map = {}
        tier_psnr_map = {}
        target_idx = 0

        for idx, prof in enumerate(self.PROFILES):
            fits_bw = prof["target_kbps"] <= (safe_bw * 1.05)
            if fits_bw:
                target_idx = idx

            tier_congestion = net_stress * (0.35 + 0.25 * idx)
            est_psnr = prof["prior_psnr"] + self.psnr_bias_per_tier[idx] - 0.14 * min(loss_pct, 25.0)

            status_map[idx] = "SAFE" if fits_bw else "OVER_BW"
            tier_congestion_map[idx] = round(tier_congestion, 3)
            tier_psnr_map[idx] = round(float(np.clip(est_psnr, 22.0, 46.0)), 2)

        # Apply CN congestion caps on high packet loss / RTT / jitter
        if loss_pct >= 8.0 or measured_rtt >= 160:
            target_idx = 0
        elif loss_pct >= 4.0 or measured_rtt >= 110 or measured_jitter >= 22.0:
            target_idx = min(target_idx, 1)
        elif loss_pct >= 1.8 or measured_rtt >= 75 or measured_jitter >= 12.0:
            target_idx = min(target_idx, 2)

        self.all_profile_status = status_map
        self.all_profile_congestion = tier_congestion_map
        self.all_profile_est_psnr = tier_psnr_map

        reason = None

        # 4. Asymmetric AIMD stability rule:
        #    - Step down immediately when congestion/bandwidth constraint is violated
        #    - Step up by at most +1 tier after 2 consecutive confirmations & 350 ms cooldown
        if target_idx < self.current_idx:
            old_name = self.PROFILES[self.current_idx]["name"].split()[0]
            self.current_idx = target_idx
            self.upward_streak = 0
            self.last_switch_time = now
            new_name = self.PROFILES[self.current_idx]["name"].split()[0]
            reason = (
                f"ABR Step Down {old_name}->{new_name} "
                f"[B_safe={int(safe_bw)}k, Loss={loss_pct:.1f}%, RTT={measured_rtt}ms]"
            )
            self.last_switch_reason = reason

        elif target_idx > self.current_idx:
            self.upward_streak += 1
            if self.upward_streak >= 2 and (now - self.last_switch_time) >= 0.35:
                old_name = self.PROFILES[self.current_idx]["name"].split()[0]
                self.current_idx = min(len(self.PROFILES) - 1, self.current_idx + 1)
                self.upward_streak = 0
                self.last_switch_time = now
                new_name = self.PROFILES[self.current_idx]["name"].split()[0]
                reason = (
                    f"ABR Step Up {old_name}->{new_name} "
                    f"[B_safe={int(safe_bw)}k, Loss={loss_pct:.1f}%]"
                )
                self.last_switch_reason = reason
        else:
            self.upward_streak = 0

        return self.get_current_profile(), reason