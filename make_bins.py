import struct
import random

def create_checkpoint(filename, layers, channels):
    Vp = 50304
    maxT = 1024
    header = [0] * 256
    header[0] = 20240326
    header[1] = 3
    header[2] = maxT
    header[3] = Vp
    header[4] = layers
    header[5] = channels // 64
    header[6] = channels
    header[7] = Vp

    num_params = (
        Vp * channels + maxT * channels +
        layers * (12 * channels * channels + 13 * channels) +
        2 * channels
    )

    print(filename, "params:", num_params)

    with open(filename, "wb") as f:
        f.write(struct.pack("<256I", *header))
        # small random init instead of zeros
        for _ in range(num_params):
            val = random.gauss(0.0, 0.02)  # same scale as original gpt2_124M
            f.write(struct.pack("f", val))

create_checkpoint("gpt2_15M.bin", 6, 384)
create_checkpoint("gpt2_30M.bin", 8, 512)
create_checkpoint("gpt2_70M.bin", 12, 640)
