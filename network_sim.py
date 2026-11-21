import time
import random
import heapq
import threading
from collections import deque

class NetworkSimulator:
    """
    Token-Bucket Bandwidth-Limited Transmission Scheduler & Network Impairment Simulator.
    Enforces:
      1. True bit-rate pacing via a token-bucket scheduler (bandwidth_kbps).
      2. Bernoulli packet drop simulation (loss_percent).
      3. Propagation delay + normal-distributed transit jitter (delay_ms, jitter_ms).
    """
    def __init__(self, send_callback):
        self.send_callback = send_callback
        self.loss_percent = 0.0
        self.delay_ms = 0
        self.jitter_ms = 0
        self.bandwidth_kbps = 6000.0

        self._lock = threading.Lock()
        self._tx_queue = deque()      # Packets waiting for bandwidth tokens
        self._delay_heap = []         # (release_timestamp, tie_breaker, payload, target)
        self._seq_tie = 0
        self._max_queue_packets = 400 # Drop-tail router buffer limit (prevents bufferbloat)
        self.dropped_by_shaper = 0
        self.running = True

        threading.Thread(target=self._scheduler_loop, daemon=True).start()

    def update_parameters(self, loss: float, delay: int, jitter: int, bw_kbps: float):
        with self._lock:
            self.loss_percent = max(0.0, min(100.0, float(loss)))
            self.delay_ms = max(0, int(delay))
            self.jitter_ms = max(0, int(jitter))
            self.bandwidth_kbps = max(100.0, float(bw_kbps))

    def push_packet(self, data: bytes, target: tuple):
        # 1. Simulate physical channel packet drop
        if self.loss_percent > 0.0:
            if random.random() * 100.0 < self.loss_percent:
                return

        with self._lock:
            # Drop-tail queue discipline if link is severely congested
            if len(self._tx_queue) >= self._max_queue_packets:
                self.dropped_by_shaper += 1
                return
            self._tx_queue.append((data, target))

    def _scheduler_loop(self):
        """
        High-precision token-bucket rate limiter + propagation delay release loop.
        Controls exact bits transmitted per second onto the UDP socket.
        """
        tokens_bytes = 4096.0
        last_time = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            with self._lock:
                bw_bps = self.bandwidth_kbps * 1000.0
                bytes_per_sec = bw_bps / 8.0
                # Max burst bucket equal to 50ms of link capacity (min 3000B for 2 MTU packets)
                max_bucket = max(3000.0, bytes_per_sec * 0.05)
                tokens_bytes = min(max_bucket, tokens_bytes + dt * bytes_per_sec)

                # Serialize packets out of the token bucket at the exact configured rate
                while self._tx_queue:
                    pkt_data, target = self._tx_queue[0]
                    pkt_len = len(pkt_data)
                    if tokens_bytes >= pkt_len:
                        tokens_bytes -= pkt_len
                        self._tx_queue.popleft()

                        # Calculate propagation delay + transit jitter
                        jitter_offset = random.gauss(0.0, self.jitter_ms / 2.0) if self.jitter_ms > 0 else 0.0
                        total_delay_sec = max(0.0, (self.delay_ms + jitter_offset) / 1000.0)
                        release_time = now + total_delay_sec

                        self._seq_tie += 1
                        heapq.heappush(self._delay_heap, (release_time, self._seq_tie, pkt_data, target))
                    else:
                        break

                # Release packets whose propagation delay deadline has arrived
                ready_packets = []
                while self._delay_heap and self._delay_heap[0][0] <= now:
                    _, _, pkt_data, target = heapq.heappop(self._delay_heap)
                    ready_packets.append((pkt_data, target))

            for pkt_data, target in ready_packets:
                self.send_callback(pkt_data, target)

            time.sleep(0.001)