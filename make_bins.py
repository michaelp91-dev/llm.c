import struct

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
    header[7] = Vp   # padded vocab
    num_params = Vp * channels + maxT * channels + layers * (12 * channels * channels + 13 * channels)
    print(filename, "target params:", num_params)
    with open(filename, "wb") as f:
        f.write(struct.pack("<256I", *header))
        # fill weights with zeros (model will train from scratch)
        f.write(b'\0' * (num_params * 4))

create_checkpoint("gpt2_15M.bin", 6, 384)
create_checkpoint("gpt2_30M.bin", 8, 512)
create_checkpoint("gpt2_70M.bin", 12, 640)
