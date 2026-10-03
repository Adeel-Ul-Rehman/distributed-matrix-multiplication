import socket
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from common import send_data, recv_data

WORKERS = [
    ("192.168.100.36", 5000),
    ("192.168.100.20", 5000),
    ("192.168.100.6", 5000),
]

MATRIX_SIZE = 1000
LABEL_W = 34


def to_mb(num_bytes):
    return num_bytes / (1024 ** 2)

def receive_result(sock):
    response = recv_data(sock)
    sock.close()
    return response

def distribute_work(A, B):
    rows = A.shape[0]
    cols = B.shape[1]

    workers_count = len(WORKERS)
    base_chunk = rows // workers_count
    remainder = rows % workers_count

    sockets = []
    worker_rows = []
    bytes_sent = []

    start = 0
    send_start_time = time.time()

    for i, worker in enumerate(WORKERS):
        extra = 1 if i < remainder else 0
        end = start + base_chunk + extra

        A_chunk = A[start:end]

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(worker)

        send_data(
            sock,
            {
                "start_row": start,
                "A_chunk": A_chunk,
                "B": B
            }
        )

        sockets.append(sock)
        worker_rows.append(A_chunk.shape[0])
        bytes_sent.append(A_chunk.nbytes + B.nbytes)

        start = end

    send_end_time = time.time()

    C = np.zeros((rows, cols), dtype=A.dtype)

    wait_start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(sockets)) as executor:
        responses = list(executor.map(receive_result, sockets))

    wait_end_time = time.time()

    worker_comp_times = []
    bytes_recv = []

    for response in responses:
        start_row = response["start_row"]
        result = response["result"]

        worker_comp_times.append(
            response.get("computation_time", 0.0)
        )

        bytes_recv.append(result.nbytes)

        end_row = start_row + result.shape[0]
        C[start_row:end_row] = result

    stats = {
        "send_time": send_end_time - send_start_time,
        "wait_and_recv_time": wait_end_time - wait_start_time,
        "worker_comp_times": worker_comp_times,
        "worker_rows": worker_rows,
        "bytes_sent": bytes_sent,
        "bytes_recv": bytes_recv,
    }

    return C, stats

def row(label, value):
    print(f"{label:<{LABEL_W}}: {value}")

if __name__ == "__main__":

    print(f"Generating {MATRIX_SIZE}x{MATRIX_SIZE} matrices...")

    A = np.random.randint(
        0,
        10,
        size=(MATRIX_SIZE, MATRIX_SIZE)
    )

    B = np.random.randint(
        0,
        10,
        size=(MATRIX_SIZE, MATRIX_SIZE)
    )

    total_start = time.time()

    C, stats = distribute_work(A, B)

    total_time = time.time() - total_start

    total_rows = A.shape[0]

    total_sent_mb = to_mb(sum(stats["bytes_sent"]))
    total_recv_mb = to_mb(sum(stats["bytes_recv"]))

    total_net_time = (stats["send_time"] + stats["wait_and_recv_time"])

    parallel_comp_time = max(stats["worker_comp_times"])

    print("\n" + "=" * 62)
    print("       DISTRIBUTED MATRIX MULTIPLICATION REPORT")
    print("=" * 62)

    print("\n--- SIZE & DISTRIBUTION ---")

    row("Matrix Size", f"{A.shape[0]} x {A.shape[1]}")
    row("Result Shape", str(C.shape))
    row("Total Data (A + B)", f"{to_mb(A.nbytes + B.nbytes):.2f} MB")
    row("Total Rows", total_rows)
    row("Master Role", "Coordinator")

    for i, (ip, port) in enumerate(WORKERS):
        row(
            f"Worker {i + 1} [{ip}] Rows",
            f"{stats['worker_rows'][i]} ({to_mb(stats['bytes_sent'][i]):.2f} MB sent)"
        )

    row("Total Sent", f"{total_sent_mb:.2f} MB")
    row("Total Received", f"{total_recv_mb:.2f} MB")

    print("\n--- EXECUTION TIME ---")

    row("Distributed Time", f"{total_time:.3f} s")
    row("Parallel Compute Time", f"{parallel_comp_time:.3f} s")
    row("Network + IO Time", f"{total_net_time:.3f} s")

    print("\n--- WORKER COMPUTATION ---")

    for i, (ip, port) in enumerate(WORKERS):
        row(
            f"Worker {i + 1} [{ip}]",
            f"{stats['worker_comp_times'][i]:.3f} s"
        )

    print("\n" + "=" * 62)