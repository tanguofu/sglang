import os
for dev in sorted(os.listdir('/sys/class/infiniband')):
    gid_file = f'/sys/class/infiniband/{dev}/ports/1/gids/3'
    try:
        gid = open(gid_file).read().strip()
        ipv4_parts = gid.split(':')[-2:]
        ip = f'{int(ipv4_parts[0][:2],16)}.{int(ipv4_parts[0][2:4],16)}.{int(ipv4_parts[1][:2],16)}.{int(ipv4_parts[1][2:4],16)}'
        print(f'{dev}: GID[3]={ip}')
    except: pass
