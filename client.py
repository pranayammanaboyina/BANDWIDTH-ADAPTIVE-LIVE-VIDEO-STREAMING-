import socket
import threading
import time
import heapq
import cv2
import numpy as np
from collections import defaultdict, deque

from rtp_packet import RTPPacket
from dip_engine import DIPProcessor

RTP_PORT = 5004
RTCP_PORT = 5005
LISTEN_IP = "0.0.0.0"


class ReorderLossTracker:
    """
    32-Packet Sliding Sequence Window for Accurate Packet Loss Measurement.
    Prevents out-of-order UDP packets from being falsely counted as lost.
    """
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.initialized = False
        self.base_seq = 0
        self.max_seq = 0
        self.received_set = set()
        self.window_received = 0
        self.window_lost = 0

    def _seq_diff(self, a: int, b: int) -> int:
        diff = (a - b) & 0xFFFF
        if diff >= 0x8000:
            diff -= 0x10000
        return diff

    def record_packet(self, seq_num: int):
        seq = seq_num & 0xFFFF
        if not self.initialized:
            self.initialized = True
            self.base_seq = seq
            self.max_seq = seq
            self.received_set.add(seq)
            self.window_received += 1
            return

        if self._seq_diff(seq, self.base_seq) < 0:
            return

        if seq not in self.received_set:
            self.received_set.add(seq)
            self.window_received += 1

        if self._seq_diff(seq, self.max_seq) > 0:
            self.max_seq = seq

        while self._seq_diff(self.max_seq, self.base_seq) >= self.window_size:
            if self.base_seq in self.received_set:
                self.received_set.remove(self.base_seq)
            else:
                self.window_lost += 1
            self.base_seq = (self.base_seq + 1) & 0xFFFF

    def harvest_interval_loss(self) -> float:
        total = self.window_received + self.window_lost
        loss_rate = float(self.window_lost / total) if total > 0 else 0.0
        self.window_received = 0
        self.window_lost = 0
        return loss_rate


class SmoothRTPReceiver:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((LISTEN_IP, RTP_PORT))
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except Exception:
            pass

        self.rtcp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtcp_sock.bind((LISTEN_IP, RTCP_PORT))
        self.server_ip = None

        self._lock = threading.Lock()
        self.frame_buffer = defaultdict(dict)
        self.frame_metadata = {}

        self.completed_frame_ids = set()
        self.completed_frame_queue = deque(maxlen=250)

        self.playout_heap = []
        self.last_played_sender_ts = -1
        self.last_played_wall_sec = None

        self.loss_tracker = ReorderLossTracker(window_size=32)
        self.last_transit = 0
        self.jitter = 1.0
        self.adaptive_jitter_buffer_ms = 25

        self.fec_recovered_frames = 0
        self.late_dropped_frames = 0
        self.concealed_frames = 0

        self.true_ref_frame = None
        self.last_true_decoded_fid = -1

        # 3-frame history buffer for Anti-Freeze Motion Synthesis
        self.flow_hist_3 = None
        self.flow_hist_2 = None
        self.flow_hist_1 = None

    def _mark_frame_completed(self, fid: int):
        if fid not in self.completed_frame_ids:
            if len(self.completed_frame_queue) == self.completed_frame_queue.maxlen:
                oldest = self.completed_frame_queue.popleft()
                self.completed_frame_ids.discard(oldest)
            self.completed_frame_queue.append(fid)
            self.completed_frame_ids.add(fid)

    def start_receiver(self):
        threading.Thread(target=self._rtp_listener_thread, daemon=True).start()
        threading.Thread(target=self._ping_pong_responder_thread, daemon=True).start()
        threading.Thread(target=self._rtcp_reporter_thread, daemon=True).start()
        self._display_loop()

    def _ping_pong_responder_thread(self):
        while True:
            try:
                data, addr = self.rtcp_sock.recvfrom(256)
                if not self.server_ip:
                    self.server_ip = addr[0]
                msg = data.decode("utf-8", errors="ignore")
                if msg.startswith("PING:"):
                    pong_msg = "PONG:" + msg[5:]
                    self.rtcp_sock.sendto(pong_msg.encode("utf-8"), (addr[0], RTCP_PORT))
            except Exception:
                pass

    def _rtp_listener_thread(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
                if not self.server_ip:
                    self.server_ip = addr[0]

                pkt = RTPPacket.decode(data)
                now_sec = time.perf_counter()
                curr_arrival_ms = int(now_sec * 1000)

                with self._lock:
                    self.loss_tracker.record_packet(pkt.seq_num)

                    transit = curr_arrival_ms - pkt.timestamp
                    if self.last_transit != 0:
                        d = abs(transit - self.last_transit)
                        self.jitter += (d - self.jitter) / 16.0
                    self.last_transit = transit

                    self.adaptive_jitter_buffer_ms = max(20, min(120, int(self.jitter * 2.0)))

                    fid = pkt.frame_id

                    if fid in self.completed_frame_ids:
                        continue

                    if self.last_played_sender_ts != -1:
                        ts_diff = (pkt.timestamp - self.last_played_sender_ts) & 0xFFFFFFFF
                        if ts_diff > 0x80000000:
                            self.late_dropped_frames += 1
                            self._mark_frame_completed(fid)
                            continue

                    self.frame_buffer[fid][pkt.frag_id] = pkt.payload

                    if fid not in self.frame_metadata:
                        video_pt = pkt.pt if pkt.pt != RTPPacket.PT_FEC else None
                        self.frame_metadata[fid] = {
                            "total_frags": pkt.total_frags,
                            "timestamp": pkt.timestamp,
                            "pt": video_pt,
                            "first_arrival_sec": now_sec,
                            "deadline_sec": now_sec + (self.adaptive_jitter_buffer_ms / 1000.0)
                        }
                    else:
                        if pkt.pt != RTPPacket.PT_FEC and self.frame_metadata[fid]["pt"] is None:
                            self.frame_metadata[fid]["pt"] = pkt.pt

                    self._try_assemble_frame(fid, now_sec)

            except Exception:
                pass

    def _try_assemble_frame(self, fid, now_sec):
        meta = self.frame_metadata.get(fid)
        if not meta:
            return

        frags = self.frame_buffer[fid]
        total_frags = meta["total_frags"]
        recovered_via_fec = False

        if not all(i in frags for i in range(total_frags)):
            if len(frags) == total_frags and total_frags in frags:
                reconstructed = RTPPacket.recover_missing_packet(frags, total_frags)
                if reconstructed is not None:
                    frags = reconstructed
                    recovered_via_fec = True

        if all(i in frags for i in range(total_frags)) and meta["pt"] is not None:
            if recovered_via_fec:
                self.fec_recovered_frames += 1

            raw_bytes = b"".join(frags[i] for i in range(total_frags))
            is_p_frame = (meta["pt"] == RTPPacket.PT_P_FRAME)

            heapq.heappush(
                self.playout_heap,
                (meta["timestamp"], fid, meta["deadline_sec"], raw_bytes, is_p_frame)
            )

            self._mark_frame_completed(fid)
            del self.frame_buffer[fid]
            del self.frame_metadata[fid]

        expired_fids = [
            f_id for f_id, m in self.frame_metadata.items()
            if now_sec > (m["deadline_sec"] + 0.120)
        ]
        for f_id in expired_fids:
            self.late_dropped_frames += 1
            self._mark_frame_completed(f_id)
            del self.frame_buffer[f_id]
            del self.frame_metadata[f_id]

    def _rtcp_reporter_thread(self):
        while True:
            time.sleep(0.3)
            if not self.server_ip:
                continue

            with self._lock:
                loss_rate = self.loss_tracker.harvest_interval_loss()
                jitter_val = self.jitter
                concealed_val = self.concealed_frames
                fec_val = self.fec_recovered_frames
                late_val = self.late_dropped_frames
                delay_val = self.adaptive_jitter_buffer_ms

            msg = (
                f"LOSS:{loss_rate:.4f};JITTER:{jitter_val:.2f};CONCEALED:{concealed_val};"
                f"FEC_FRAMES:{fec_val};LATE:{late_val};DELAY:{delay_val}"
            )
            try:
                self.rtcp_sock.sendto(msg.encode("utf-8"), (self.server_ip, RTCP_PORT))
            except Exception:
                pass

    def _push_flow_history(self, frame_bgr: np.ndarray):
        self.flow_hist_3 = self.flow_hist_2
        self.flow_hist_2 = self.flow_hist_1
        self.flow_hist_1 = frame_bgr.copy()

    def _display_loop(self):
        cv2.namedWindow("Laptop 2 - Custom RTP-Inspired Video Player", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Laptop 2 - Custom RTP-Inspired Video Player", 1280, 720)

        while True:
            loop_start = time.perf_counter()
            display_frame = None
            selected_frame = None
            trigger_concealment = False
            is_concealed_frame = False

            with self._lock:
                while self.playout_heap and self.last_played_sender_ts != -1:
                    top_ts = self.playout_heap[0][0]
                    ts_diff = (top_ts - self.last_played_sender_ts) & 0xFFFFFFFF
                    if ts_diff > 0x80000000:
                        heapq.heappop(self.playout_heap)
                        self.late_dropped_frames += 1
                    else:
                        break

                if self.playout_heap:
                    top_ts, fid, deadline_sec, raw_bytes, is_p_frame = self.playout_heap[0]
                    if loop_start >= deadline_sec or len(self.playout_heap) >= 2:
                        selected_frame = heapq.heappop(self.playout_heap)
                        self.last_played_sender_ts = top_ts
                        self.last_played_wall_sec = loop_start
                else:
                    if self.last_played_wall_sec is not None:
                        overdue_threshold = (
                            self.last_played_wall_sec
                            + 0.085
                            + (self.adaptive_jitter_buffer_ms / 1000.0)
                        )
                        if loop_start >= overdue_threshold:
                            trigger_concealment = True
                            self.last_played_wall_sec = loop_start

            if selected_frame is not None:
                _, fid, _, raw_bytes, is_p_frame = selected_frame

                if not is_p_frame:
                    decoded = DIPProcessor.decode_frame(raw_bytes, False, None)
                    if decoded is not None:
                        display_frame = decoded
                        self.true_ref_frame = decoded.copy()
                        self.last_true_decoded_fid = fid
                        self._push_flow_history(decoded)
                else:
                    expected_fid = (self.last_true_decoded_fid + 1) & 0xFFFFFFFF
                    if self.true_ref_frame is not None and fid == expected_fid:
                        decoded = DIPProcessor.decode_frame(raw_bytes, True, self.true_ref_frame)
                        if decoded is not None:
                            display_frame = decoded
                            self.true_ref_frame = decoded.copy()
                            self.last_true_decoded_fid = fid
                            self._push_flow_history(decoded)

            elif trigger_concealment:
                if self.flow_hist_1 is not None and self.flow_hist_2 is not None:
                    synthesized = DIPProcessor.synthesize_missing_frame(
                        self.flow_hist_2, self.flow_hist_1, self.flow_hist_3
                    )
                    if synthesized is not None:
                        display_frame = synthesized
                        self._push_flow_history(synthesized)
                        self.concealed_frames += 1
                        is_concealed_frame = True

            if display_frame is not None:
                h_in, w_in = display_frame.shape[:2]
                if w_in <= 640:
                    dl_tag = "ON (AR-ResNet CNN)" if DIPProcessor.is_dl_active() else "ON (Bilateral)"
                else:
                    dl_tag = "BYPASS (HD)"

                mode_tag = "ANTI-FREEZE SYNTHESIS" if is_concealed_frame else "LIVE STREAM"
                canvas = cv2.resize(display_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                hud_text = (
                    f"{mode_tag} | {w_in}x{h_in} | DL RESTORE: {dl_tag} | "
                    f"JITTER BUF: {self.adaptive_jitter_buffer_ms}ms | FEC: {self.fec_recovered_frames} | "
                    f"SYNTHESIZED: {self.concealed_frames}"
                )
                color = (0, 215, 255) if is_concealed_frame else (0, 255, 0)
                cv2.putText(canvas, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2)
                cv2.imshow("Laptop 2 - Custom RTP-Inspired Video Player", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - loop_start
            if 0.025 > elapsed:
                time.sleep(0.025 - elapsed)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    receiver = SmoothRTPReceiver()
    receiver.start_receiver()