import socket
import time
from common import recv_data, send_data, multiply_chunk

HOST = "10.182.127.221"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"Worker listening on {HOST}:{PORT}")

while True:
    conn, addr = server.accept()
    print("Connected by", addr)

    try:
        task = recv_data(conn)

        start_row = task["start_row"]
        A_chunk = task["A_chunk"]
        B = task["B"]

        start_time = time.time()
        result = multiply_chunk(A_chunk, B)
        computation_time = time.time() - start_time

        send_data(
            conn,
            {
                "start_row": start_row,
                "result": result,
                "computation_time": computation_time
            }
        )

        print(f"Processed rows starting at {start_row}")

    except Exception as error:
        print("Error:", error)
    finally:
        conn.close()