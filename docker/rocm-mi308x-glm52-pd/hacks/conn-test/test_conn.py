import socket
for ip, port, label in [
    ('29.199.73.122', 16435, 'RDMA IP'),
    ('21.234.171.87', 16435, 'Mgmt IP'),
]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect((ip, port))
        print(f'{label} {ip}:{port} - REACHABLE')
        s.close()
    except Exception as e:
        print(f'{label} {ip}:{port} - FAILED: {e}')
