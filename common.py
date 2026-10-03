import pickle
import struct
import numpy as np


def send_data(sock, data):
    serialized = pickle.dumps(data)
    sock.sendall(struct.pack("!I", len(serialized)))
    sock.sendall(serialized)

def recv_data(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None

    data_len = struct.unpack("!I", raw_len)[0]
    data = recvall(sock, data_len)
    return pickle.loads(data)

def recvall(sock, n):
    data = b''

    while len(data) < n:
        packet = sock.recv(n - len(data))

        if not packet:
            return None

        data += packet

    return data

def multiply_chunk(A_chunk, B):
    return np.dot(A_chunk, B)