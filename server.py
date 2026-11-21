import socket
import threading
import time
import cv2
import csv
import io
import argparse
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, make_response

from rtp_packet import RTPPacket
from network_sim import NetworkSimulator
from dip_engine import DIPProcessor, _cnn_restorer
from controller import AdaptiveBitrateController

parser = argparse.ArgumentParser(description="Adaptive UDP Streaming Hub")
parser.add_argument("--client", default="192.168.137.204", help="Target Client IP")
args, _ = parser.parse_known_args()

CLIENT_IP = args.client
RTP_PORT = 5004
RTCP_PORT = 5005
MTU = 1200

app = Flask(__name__)


class MasterSenderSystem:
    def __init__(self, target_ip):
        self.target_ip = target_ip

        self.raw_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except Exception:
            pass

        self.rtcp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtcp_sock.bind(("0.0.0.0", RTCP_PORT))

        self.net_sim = NetworkSimulator(self._raw_udp_send)
        self.controller = AdaptiveBitrateController()

        self.streaming_active = True
        self.view_mode = "normal"
        self.frame_id = 0
        self.seq_num = 0

        self.achieved_throughput_kbps = 1400.0
        self.configured_bw_limit_kbps = 6000.0
        self.rtcp_measured_loss_frac = 0.0
        self.measured_rtt_ms = 8
        self.real_wifi_jitter = 1.0
        self.client_jitter_buf_ms = 25
        self.fec_recovered_frames = 0
        self.late_dropped_frames = 0
        self.concealed_frames = 0

        self.mse = 12.0
        self.encoder_psnr = 40.2
        self.ssim = 0.968

        self.total_bytes_sent = 0
        self.cached_display = None
        self.bytes_sent_window = 0
        self.last_bitrate_calc = time.time()
        self.last_eval_time = time.time()

        self.logs = []
        self.switch_events = []
        self.sessions = [
            {
                "id": "EXP-01",
                "name": "Custom RTP/UDP + XOR FEC + Jitter Buffer",
                "avg_loss": "32-Pkt Sliding Window",
                "avg_psnr": "Measured Codec PSNR",
                "status": "Active"
            },
            {
                "id": "EXP-02",
                "name": "ROI 8x8 DCT + AR-ResNet DL + Anti-Freeze Flow",
                "avg_loss": "Zero-Freeze Conceal",
                "avg_psnr": "L_Stream Perceptual",
                "status": "Active"
            }
        ]

    @staticmethod
    def _sender_clock_ms() -> int:
        return int(time.perf_counter() * 1000) & 0xFFFFFFFF

    def _raw_udp_send(self, data: bytes, target: tuple):
        try:
            self.raw_sock.sendto(data, target)
            self.bytes_sent_window += len(data)
            self.total_bytes_sent += len(data)
        except Exception:
            pass

    def rtt_probe_sender_thread(self):
        while True:
            time.sleep(0.25)
            if not self.target_ip:
                continue
            ping_ts = self._sender_clock_ms()
            ping_msg = f"PING:{ping_ts}".encode("utf-8")
            try:
                self.rtcp_sock.sendto(ping_msg, (self.target_ip, RTCP_PORT))
            except Exception:
                pass

    def rtcp_listener(self):
        self.rtcp_sock.settimeout(0.5)
        while True:
            try:
                data, addr = self.rtcp_sock.recvfrom(512)
                if self.target_ip != addr[0]:
                    self.target_ip = addr[0]

                msg = data.decode("utf-8", errors="ignore")

                if msg.startswith("PONG:"):
                    echoed_ts = int(msg[5:])
                    now_ms = self._sender_clock_ms()
                    rtt_sample = (now_ms - echoed_ts) & 0xFFFFFFFF
                    if 0 < rtt_sample < 2000:
                        self.measured_rtt_ms = max(1, int(0.65 * self.measured_rtt_ms + 0.35 * rtt_sample))
                    continue

                parts = dict(item.split(":") for item in msg.split(";"))
                self.rtcp_measured_loss_frac = max(0.0, min(1.0, float(parts.get("LOSS", 0.0))))
                self.real_wifi_jitter = float(parts.get("JITTER", 1.0))
                self.concealed_frames = int(parts.get("CONCEALED", self.concealed_frames))
                self.fec_recovered_frames = int(parts.get("FEC_FRAMES", self.fec_recovered_frames))
                self.late_dropped_frames = int(parts.get("LATE", self.late_dropped_frames))
                self.client_jitter_buf_ms = int(parts.get("DELAY", self.client_jitter_buf_ms))
            except socket.timeout:
                continue
            except Exception:
                pass

    def capture_and_stream_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_reconstructed_frame = None
        last_profile_idx = self.controller.current_idx
        gop_counter = 0
        GOP_SIZE = 8

        while True:
            if not self.streaming_active:
                time.sleep(0.033)
                continue

            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            if frame.shape[1] < 1280 or frame.shape[0] < 720:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)

            if loop_start - self.last_eval_time >= 0.3:
                self.last_eval_time = loop_start

                sim_jitter = self.net_sim.jitter_ms
                sim_delay = self.net_sim.delay_ms
                self.configured_bw_limit_kbps = float(self.net_sim.bandwidth_kbps)

                measured_loss_frac = self.rtcp_measured_loss_frac
                effective_rtt = max(self.measured_rtt_ms, int(sim_delay * 2))
                measured_jitter = max(self.real_wifi_jitter, float(sim_jitter))

                profile, reason = self.controller.evaluate(
                    measured_loss=measured_loss_frac,
                    measured_rtt=effective_rtt,
                    measured_jitter=measured_jitter,
                    configured_bw_kbps=self.configured_bw_limit_kbps,
                    achieved_kbps=self.achieved_throughput_kbps
                )

                if reason and reason not in [e['reason'] for e in self.switch_events[-2:]]:
                    self.switch_events.append({
                        "time": time.strftime("%H:%M:%S"),
                        "reason": reason,
                        "res": profile['res'],
                        "fps": profile['fps'],
                        "q": profile['q']
                    })

            profile = self.controller.get_current_profile()
            profile_changed = (self.controller.current_idx != last_profile_idx)
            last_profile_idx = self.controller.current_idx

            self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
            ts = self._sender_clock_ms()

            active_loss_indicator = max(
                self.rtcp_measured_loss_frac * 100.0,
                self.net_sim.loss_percent
            )

            force_keyframe = (gop_counter % GOP_SIZE == 0) or profile_changed
            gop_counter += 1

            compressed_bytes, downscaled_orig, is_p_frame = DIPProcessor.process_frame(
                frame,
                profile['res'],
                profile['q'],
                prev_frame=prev_reconstructed_frame,
                force_keyframe=force_keyframe
            )

            decoded_current = DIPProcessor.decode_frame(
                compressed_bytes,
                is_p_frame,
                prev_frame=prev_reconstructed_frame
            )
            if decoded_current is None:
                decoded_current = downscaled_orig.copy()

            self.mse, self.encoder_psnr, self.ssim = DIPProcessor.calculate_all_metrics(
                downscaled_orig, decoded_current
            )
            prev_reconstructed_frame = decoded_current.copy()

            if self.view_mode == 'diff':
                diff_img = DIPProcessor.generate_diff_map(downscaled_orig, decoded_current)
                _, buf = cv2.imencode('.jpg', diff_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                self.cached_display = buf.tobytes()
            else:
                _, buf = cv2.imencode('.jpg', decoded_current, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                self.cached_display = buf.tobytes()

            total_size = len(compressed_bytes)
            chunks = [compressed_bytes[i:i + MTU] for i in range(0, total_size, MTU)]
            num_frags = len(chunks)
            payload_type = RTPPacket.PT_P_FRAME if is_p_frame else RTPPacket.PT_I_FRAME

            for frag_id, chunk in enumerate(chunks):
                self.seq_num = (self.seq_num + 1) & 0xFFFF
                pkt = RTPPacket(
                    seq_num=self.seq_num,
                    timestamp=ts,
                    frame_id=self.frame_id,
                    frag_id=frag_id,
                    total_frags=num_frags,
                    payload=chunk,
                    pt=payload_type
                )
                self.net_sim.push_packet(pkt.encode(), (self.target_ip, RTP_PORT))

            if num_frags >= 1 and active_loss_indicator > 0.5:
                self.seq_num = (self.seq_num + 1) & 0xFFFF
                parity_payload = RTPPacket.generate_xor_parity(chunks)
                fec_pkt = RTPPacket(
                    seq_num=self.seq_num,
                    timestamp=ts,
                    frame_id=self.frame_id,
                    frag_id=num_frags,
                    total_frags=num_frags,
                    payload=parity_payload,
                    pt=RTPPacket.PT_FEC
                )
                self.net_sim.push_packet(fec_pkt.encode(), (self.target_ip, RTP_PORT))

            now = time.time()
            if now - self.last_bitrate_calc >= 0.5:
                dt = now - self.last_bitrate_calc
                self.achieved_throughput_kbps = (self.bytes_sent_window * 8.0) / (dt * 1000.0)
                self.bytes_sent_window = 0
                self.last_bitrate_calc = now

                eff_rtt = max(self.measured_rtt_ms, int(self.net_sim.delay_ms * 2))
                eff_jitter = max(self.real_wifi_jitter, float(self.net_sim.jitter_ms))
                feedback_loss_pct = round(self.rtcp_measured_loss_frac * 100.0, 2)

                frame_type_str = "P-FRAME (DELTA)" if is_p_frame else "I-FRAME (KEY)"
                self.logs.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "achieved_throughput_kbps": round(self.achieved_throughput_kbps, 1),
                    "configured_bw_limit_kbps": round(self.configured_bw_limit_kbps, 1),
                    "feedback_measured_loss_pct": feedback_loss_pct,
                    "configured_sim_loss_pct": round(self.net_sim.loss_percent, 2),
                    "measured_encoder_psnr_db": round(self.encoder_psnr, 2),
                    "resolution": profile['res'],
                    "fps": 30,
                    "quality": profile['q'],
                    "ping_pong_rtt_ms": eff_rtt,
                    "jitter_ms": round(eff_jitter, 2),
                    "ssim": round(self.ssim, 3),
                    "mse": round(self.mse, 2),
                    "type": frame_type_str,
                    "fec_recovered_frames": self.fec_recovered_frames,
                    "concealed_frames": self.concealed_frames,
                    "late_dropped_frames": self.late_dropped_frames,
                    "jitter_buffer_ms": self.client_jitter_buf_ms,
                    "roi_ratio": DIPProcessor.last_roi_ratio,
                    "edge_sharpness": DIPProcessor.last_edge_sharpness,
                    "fft_hf_ratio": DIPProcessor.last_fft_hf_ratio
                })
                if len(self.logs) > 600:
                    self.logs.pop(0)

            elapsed = time.time() - loop_start
            target_delay = 1.0 / 30.0
            if target_delay > elapsed:
                time.sleep(target_delay - elapsed)


sender = MasterSenderSystem(CLIENT_IP)


@app.route('/')
def index():
    return render_template('index.html')


def gen_display_stream():
    while True:
        buf = sender.cached_display
        if buf is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf + b'\r\n')
        time.sleep(0.033)


@app.route('/feed_stream')
def feed_stream():
    return Response(gen_display_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/telemetry')
def telemetry():
    prof = sender.controller.get_current_profile()
    active_q = prof['q']
    quant_matrix = DIPProcessor.get_scaled_quant_matrix(active_q)

    measured_loss = round(sender.rtcp_measured_loss_frac * 100.0, 1)
    effective_rtt = max(sender.measured_rtt_ms, int(sender.net_sim.delay_ms * 2))
    effective_jitter = max(round(sender.real_wifi_jitter, 1), round(sender.net_sim.jitter_ms, 1))
    total_data_mb = round(sender.total_bytes_sent / (1024 * 1024), 2)

    est_drop_risk = round(min(99.0, sender.controller.last_net_stress * 100.0), 1)
    dl_gain = 1.35 if (active_q <= 65 and DIPProcessor.is_dl_active()) else 0.0
    est_restored_psnr = round(min(48.0, sender.encoder_psnr + dl_gain), 1)
    dl_status = (
        f"AR-ResNet Active ({_cnn_restorer.loss_type}={_cnn_restorer.last_loss_value})"
        if DIPProcessor.is_dl_active() else "AR-ResNet Ready"
    )
    live_samples = len(sender.controller.pred_psnr_errors)
    val_stats = sender.controller.get_validation_stats()

    return jsonify({
        'status': 'ONLINE' if sender.streaming_active else 'PAUSED',
        'client_ip': sender.target_ip,
        'resolution': prof['res'],
        'fps': 30,
        'quality': active_q,
        'profile_id': sender.controller.current_idx,
        'profile_name': prof['name'],
        'view_mode': sender.view_mode,
        'achieved_throughput_kbps': round(sender.achieved_throughput_kbps, 1),
        'configured_bw_limit_kbps': round(sender.configured_bw_limit_kbps, 1),
        'loss_rate': measured_loss,
        'configured_sim_loss_pct': round(sender.net_sim.loss_percent, 1),
        'rtt': effective_rtt,
        'jitter': effective_jitter,
        'total_data_mb': total_data_mb,
        'jitter_buf_ms': sender.client_jitter_buf_ms,
        'fec_recovered_frames': sender.fec_recovered_frames,
        'late_dropped': sender.late_dropped_frames,
        'concealed_frames': sender.concealed_frames,
        'encoder_psnr': round(sender.encoder_psnr, 1),
        'ssim': round(sender.ssim, 3),
        'mse': round(sender.mse, 2),
        'frame_id': sender.frame_id,
        'reason': sender.controller.last_switch_reason,
        'events': sender.switch_events[-8:],
        'sessions': sender.sessions,
        'quant_matrix': quant_matrix,
        'roi_ratio': DIPProcessor.last_roi_ratio,
        'edge_sharpness': DIPProcessor.last_edge_sharpness,
        'fft_hf_ratio': DIPProcessor.last_fft_hf_ratio,
        # Standard & Custom DL Loss Function Telemetry:
        'loss_type': _cnn_restorer.loss_type if _cnn_restorer else 'none',
        'loss_total': _cnn_restorer.last_loss_value if _cnn_restorer else 0.0,
        'loss_breakdown': _cnn_restorer.loss_breakdown if _cnn_restorer else {},
        # Dashboard KPI card keys:
        'ai_frame_loss_prob_pct': est_drop_risk,
        'ai_predicted_codec_psnr_db': est_restored_psnr,
        'ai_live_samples': live_samples,
        'ai_loss_accuracy_pct': val_stats['hit_rate_pct'],
        'ai_psnr_mae_db': val_stats['psnr_mae'],
        'ai_psnr_rmse_db': val_stats['psnr_rmse'],
        'ai_validation_status': dl_status
    })


@app.route('/control/<action>')
def control_action(action):
    if action == 'toggle_stream':
        sender.streaming_active = not sender.streaming_active
    elif action == 'toggle_view':
        sender.view_mode = 'diff' if sender.view_mode == 'normal' else 'normal'
    return jsonify({'status': 'ok'})


@app.route('/network_lab/apply')
def apply_net_lab():
    loss = float(request.args.get('loss', 0))
    delay = int(request.args.get('delay', 0))
    jitter = int(request.args.get('jitter', 0))
    bw = int(request.args.get('bw', 6000))
    sender.net_sim.update_parameters(loss, delay, jitter, bw)

    sender.last_eval_time = 0
    sender.controller.last_switch_time = 0

    effective_loss_frac = max(sender.rtcp_measured_loss_frac, loss / 100.0)
    measured_rtt = max(sender.measured_rtt_ms, int(delay * 2))
    measured_jitter = max(sender.real_wifi_jitter, float(jitter))
    sender.configured_bw_limit_kbps = float(bw)

    sender.controller.evaluate(
        measured_loss=effective_loss_frac,
        measured_rtt=measured_rtt,
        measured_jitter=measured_jitter,
        configured_bw_kbps=sender.configured_bw_limit_kbps,
        achieved_kbps=sender.achieved_throughput_kbps
    )
    return jsonify({'status': 'ok', 'applied_loss': loss})


@app.route('/export/csv')
def export_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "achieved_throughput_kbps", "configured_bw_limit_kbps",
        "feedback_measured_loss_pct", "configured_sim_loss_pct",
        "measured_encoder_psnr_db",
        "resolution", "fps", "quality", "ping_pong_rtt_ms", "jitter_ms", "ssim", "mse", "type",
        "fec_recovered_frames", "concealed_frames", "late_dropped_frames", "jitter_buffer_ms",
        "roi_ratio", "edge_sharpness", "fft_hf_ratio"
    ])
    writer.writeheader()
    writer.writerows(sender.logs)
    resp = make_response(output.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=streaming_telemetry.csv"
    resp.headers["Content-type"] = "text/csv"
    return resp


if __name__ == '__main__':
    threading.Thread(target=sender.rtcp_listener, daemon=True).start()
    threading.Thread(target=sender.rtt_probe_sender_thread, daemon=True).start()
    threading.Thread(target=sender.capture_and_stream_loop, daemon=True).start()
    print("==================================================================")
    print(f"  SERVER ACTIVE: TARGET RECEIVER -> {CLIENT_IP}")
    print("  CN (RTP/UDP + FEC) + DIP (ROI 8x8 DCT) + DL (AR-RESNET)")
    print("  PRODUCTION DASHBOARD: http://127.0.0.1:5000")
    print("==================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)