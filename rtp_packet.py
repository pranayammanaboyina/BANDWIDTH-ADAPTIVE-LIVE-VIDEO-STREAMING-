import struct

class RTPPacket:
    """
    Custom RTP Packetizer with True Bitwise XOR Forward Error Correction (FEC).
    Header Format (16 bytes, Big-Endian):
      - Version/Flags (1B): 0x80
      - Payload Type  (1B): 96 (I-Frame), 97 (P-Frame), 98 (XOR FEC Parity)
      - Sequence Num  (2B): 0..65535
      - Timestamp     (4B): Millisecond capture timestamp
      - Frame ID      (4B): Monotonic video frame counter
      - Frag ID       (2B): Slice index (0..total_frags-1; total_frags for FEC)
      - Total Frags   (2B): Number of data slices in this frame
    """
    HEADER_FORMAT = "!BBHIIHH"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    PT_I_FRAME = 96
    PT_P_FRAME = 97
    PT_FEC = 98

    def __init__(self, seq_num: int, timestamp: int, frame_id: int,
                 frag_id: int, total_frags: int, payload: bytes, pt: int = PT_I_FRAME):
        self.pt = pt
        self.seq_num = seq_num & 0xFFFF
        self.timestamp = timestamp & 0xFFFFFFFF
        self.frame_id = frame_id & 0xFFFFFFFF
        self.frag_id = frag_id & 0xFFFF
        self.total_frags = total_frags & 0xFFFF
        self.payload = payload

    def encode(self) -> bytes:
        header = struct.pack(
            self.HEADER_FORMAT,
            0x80,
            self.pt,
            self.seq_num,
            self.timestamp,
            self.frame_id,
            self.frag_id,
            self.total_frags
        )
        return header + self.payload

    @classmethod
    def decode(cls, raw_bytes: bytes):
        if len(raw_bytes) < cls.HEADER_SIZE:
            raise ValueError("Packet shorter than RTP header")
        _, pt, seq_num, timestamp, frame_id, frag_id, total_frags = struct.unpack(
            cls.HEADER_FORMAT, raw_bytes[:cls.HEADER_SIZE]
        )
        payload = raw_bytes[cls.HEADER_SIZE:]
        return cls(seq_num, timestamp, frame_id, frag_id, total_frags, payload, pt)

    @staticmethod
    def generate_xor_parity(chunks: list) -> bytes:
        """
        Generates bitwise XOR parity across all N data chunks of a frame.
        Prepends a 2-byte big-endian total-length XOR checksum so the exact
        byte length of any single missing fragment is preserved upon recovery.
        """
        if not chunks:
            return b""
        max_len = max(len(c) for c in chunks)
        parity = bytearray(max_len)
        xor_lengths = 0

        for chunk in chunks:
            xor_lengths ^= len(chunk)
            for idx, b in enumerate(chunk):
                parity[idx] ^= b

        return struct.pack("!H", xor_lengths) + bytes(parity)

    @staticmethod
    def recover_missing_packet(frags_dict: dict, total_frags: int):
        """
        Recovers a single missing data fragment (0 .. total_frags - 1) using
        the XOR parity packet stored at key `total_frags`.
        Returns a complete dict of {0: bytes, ..., total_frags-1: bytes} or None.
        """
        if total_frags not in frags_dict:
            return None

        missing_indices = [i for i in range(total_frags) if i not in frags_dict]
        if len(missing_indices) != 1:
            return None

        missing_idx = missing_indices[0]
        fec_raw = frags_dict[total_frags]
        if len(fec_raw) < 2:
            return None

        expected_xor_len = struct.unpack("!H", fec_raw[:2])[0]
        recovered_buf = bytearray(fec_raw[2:])
        recovered_len = expected_xor_len

        for i in range(total_frags):
            if i == missing_idx:
                continue
            chunk = frags_dict[i]
            recovered_len ^= len(chunk)
            for idx, b in enumerate(chunk):
                if idx < len(recovered_buf):
                    recovered_buf[idx] ^= b

        if recovered_len <= 0 or recovered_len > len(recovered_buf):
            return None

        result = dict(frags_dict)
        result[missing_idx] = bytes(recovered_buf[:recovered_len])
        return result