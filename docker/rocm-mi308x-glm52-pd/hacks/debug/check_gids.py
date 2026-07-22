import os
for dev in sorted(os.listdir('/sys/class/infiniband')):
    print(f'=== {dev} ===')
    ports = sorted(os.listdir(f'/sys/class/infiniband/{dev}/ports'))
    for port in ports:
        gids_dir = f'/sys/class/infiniband/{dev}/ports/{port}/gids'
        if os.path.isdir(gids_dir):
            for gid_file in sorted(os.listdir(gids_dir), key=lambda x: int(x)):
                gid = open(f'{gids_dir}/{gid_file}').read().strip()
                idx = int(gid_file)
                types_dir = f'/sys/class/infiniband/{dev}/ports/{port}/gid_attrs/types'
                type_path = f'{types_dir}/{gid_file}'
                gid_type = open(type_path).read().strip() if os.path.exists(type_path) else 'unknown'
                print(f'  GID[{idx}] = {gid}  type={gid_type}')
