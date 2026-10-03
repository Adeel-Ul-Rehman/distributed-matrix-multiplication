# Distributed Matrix Multiplication using TCP Sockets

A lightweight, robust Master-Worker distributed computing system implemented in Python. It splits large-scale matrix multiplications across networked worker nodes using raw TCP sockets, computes parallel row-wise dot products via NumPy, and benchmarks distributed execution against single-machine performance.

---

## 📌 Architecture Overview

The system follows a classic **Master-Worker (Coordinator-Worker)** distributed computing pattern:

```
                          +------------------------+
                          |   Master / Coordinator |
                          |       (master.py)      |
                          +-----------+------------+
                                      |
       +------------------------------+------------------------------+
       | Distributes Row Chunks (A_i) | Distributes Row Chunks (A_j) | Distributes Row Chunks (A_k)
       | + Matrix B via TCP Sockets   | + Matrix B via TCP Sockets   | + Matrix B via TCP Sockets
       v                              v                              v
+---------------+              +---------------+              +---------------+
|  Worker Node 1|              |  Worker Node 2|              |  Worker Node 3|
|  (worker.py)  |              |  (worker.py)  |              |  (worker.py)  |
|  Port: 5000   |              |  Port: 5000   |              |  Port: 5000   |
+-------+-------+              +-------+-------+              +-------+-------+
        |                              |                              |
        | Computes np.dot(A_chunk, B)  | Computes np.dot(A_chunk, B)  | Computes np.dot(A_chunk, B)
        v                              v                              v
        +------------------------------+------------------------------+
                                      |
                                      | Transmits partial result matrices back
                                      v
                          +------------------------+
                          |   Master / Coordinator |
                          |  Reassembles Matrix C  |
                          |  & Generates Report    |
                          +------------------------+
```

### Key Workflow
1. **Matrix Generation**: The master node generates two $N \times N$ matrices ($A$ and $B$).
2. **Row-Chunk Partitioning**: Matrix $A$ is sliced horizontally into sub-matrices ($A_1, A_2, \dots, A_k$) according to the number of active workers.
3. **Dispatch over TCP**: Sockets connect to each worker to transmit the full matrix $B$ and worker-assigned row slice $A_i$.
4. **Parallel Computation**: Each worker calculates $C_i = A_i \cdot B$ using NumPy's optimized BLAS routines and records computation time.
5. **Concurrent Gathering**: The master leverages `concurrent.futures.ThreadPoolExecutor` to asynchronously receive and stitch all partial results into final matrix $C$.
6. **Detailed Benchmarking**: Outputs a structured report comparing **Distributed Time**, **Parallel Compute Time**, and **Network + I/O Overhead**.

---

## 📡 Custom Binary Socket Protocol

TCP is a stream-oriented protocol without native message boundaries. To prevent packet fragmentation and data corruption during large NumPy array transfers, [common.py](common.py) implements a **Length-Prefixed Framing Protocol**:

- **Header**: 4-byte big-endian unsigned integer (`struct.pack("!I", len(data))`) representing the payload size in bytes.
- **Payload**: Serialized Python dictionary containing matrix chunks, metadata, and computation times (via `pickle`).
- **Receiver (`recvall`)**: Repeatedly reads from the socket until the exact expected byte length is accumulated.

---

## 📂 Project Structure

```
distributed-matrix-multiplication/
├── common.py           # Shared networking protocol (framing, send_data, recv_data) and multiply helper
├── master.py           # Coordinator: data partitioning, worker communication, reassembly, reporting
├── worker.py           # Worker daemon: listens on socket, performs matrix multiplication, returns result
├── single_system.py    # Baseline script for single-machine matrix multiplication comparison
├── requirements.txt    # Python dependencies (NumPy)
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation and instructions
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
* **Python**: Version 3.8 or higher
* **NumPy**: For matrix operations

### 2. Install Dependencies
Clone the repository and install dependencies on all machines:
```bash
git clone https://github.com/Adeel-Ul-Rehman/distributed-matrix-multiplication.git
cd distributed-matrix-multiplication
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Option A: Running on Multiple Machines (LAN / Cluster)

#### Step 1: Find Worker IP Addresses
On each worker machine, obtain its local IP address:
* **Windows**: Run `ipconfig` (look for *IPv4 Address*, e.g., `192.168.100.36`)
* **Linux/macOS**: Run `ip a` or `ifconfig`

Ensure incoming connections on port `5000` are allowed in the firewall.

#### Step 2: Configure Worker Script
In [worker.py](worker.py), set `HOST` to the worker's own local IP or `"0.0.0.0"` (to listen on all interfaces):
```python
HOST = "0.0.0.0"  # or specific IP: e.g. "192.168.100.36"
PORT = 5000
```
Start the worker:
```bash
python worker.py
```
Output:
```
Worker listening on 0.0.0.0:5000
```

#### Step 3: Configure Master Script
In [master.py](master.py), update the `WORKERS` list with your worker IPs:
```python
WORKERS = [
    ("192.168.100.36", 5000),
    ("192.168.100.20", 5000),
    ("192.168.100.6", 5000),
]
```

#### Step 4: Run the Master Node
Execute the coordinator on the master machine:
```bash
python master.py
```

---

### Option B: Local Testing on a Single Machine

You can simulate a distributed cluster locally by binding workers to distinct ports on `localhost` (`127.0.0.1`):

1. **Start Worker 1 (Terminal 1)**:
   ```bash
   python -c "import worker; worker.HOST='127.0.0.1'; worker.PORT=5001; worker.server.bind(('127.0.0.1', 5001)); worker.server.listen(1); print('Worker 1 on 5001'); exec(open('worker.py').read())"
   ```
   *(Or simply run multiple copies of `worker.py` configured with ports 5001, 5002, 5003)*

2. **Configure `master.py` for localhost**:
   ```python
   WORKERS = [
       ("127.0.0.1", 5001),
       ("127.0.0.1", 5002),
       ("127.0.0.1", 5003),
   ]
   ```

3. **Run `master.py` (Terminal 4)**:
   ```bash
   python master.py
   ```

---

### Option C: Baseline Single-Node Execution

To compare the distributed performance against a standalone single-machine run:
```bash
python single_system.py
```

---

## 📊 Sample Output & Report

When `master.py` executes successfully, it displays a comprehensive breakdown:

```text
Generating 1000x1000 matrices...

==============================================================
       DISTRIBUTED MATRIX MULTIPLICATION REPORT
==============================================================

--- SIZE & DISTRIBUTION ---
Matrix Size                       : 1000 x 1000
Result Shape                      : (1000, 1000)
Total Data (A + B)                : 15.26 MB
Total Rows                        : 1000
Master Role                       : Coordinator
Worker 1 [192.168.100.36] Rows   : 334 (10.18 MB sent)
Worker 2 [192.168.100.20] Rows   : 333 (10.17 MB sent)
Worker 3 [192.168.100.6] Rows    : 333 (10.17 MB sent)
Total Sent                        : 30.52 MB
Total Received                    : 7.63 MB

--- EXECUTION TIME ---
Distributed Time                  : 0.842 s
Parallel Compute Time             : 0.124 s
Network + IO Time                 : 0.718 s

--- WORKER COMPUTATION ---
Worker 1 [192.168.100.36]         : 0.124 s
Worker 2 [192.168.100.20]         : 0.119 s
Worker 3 [192.168.100.6]          : 0.121 s

==============================================================
```

---

## 📈 Performance Analysis & Insights

- **Compute vs. Network Bottleneck**: As shown in the report, total wall-clock time in small matrix sizes ($N \le 1000$) is dominated by network serialization and socket transmission (`Network + IO Time`).
- **Scaling Threshold**: For very large matrices ($N \ge 4000+$), $O(N^3)$ computational complexity rapidly outpaces the $O(N^2)$ network communication overhead, demonstrating noticeable speedups when scaling out across worker nodes.
- **Load Balancing**: The master partitions row counts evenly using modulo arithmetic (`base_chunk = rows // workers_count`), preventing worker stragglers.

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ConnectionRefusedError: [WinError 10061]` | Worker is not running or port is incorrect | Ensure `worker.py` is started before `master.py` and verifying IP/port configurations. |
| `WinError 10048 (Address already in use)` | Previous socket instance did not close | Wait a few seconds or terminate lingering Python processes from Task Manager / `kill`. |
| Connection times out across machines | Windows Firewall is blocking inbound TCP traffic on port 5000 | Add an Inbound Rule in Windows Defender Firewall allowing TCP traffic on port `5000`. |
| `MemoryError` | Matrix dimensions too large for system RAM | Reduce `MATRIX_SIZE` in `master.py` (e.g. from 4000 to 1000 or 2000). |

---

## 📜 License
This project was developed as part of the Parallel and Distributed Computing (PDC) coursework. Free to use for educational and research purposes.
