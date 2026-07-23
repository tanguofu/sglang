echo "########## /etc/fstab ##########"
cat /etc/fstab 2>&1
echo
echo "########## lsblk ##########"
lsblk -f -o NAME,FSTYPE,LABEL,MOUNTPOINT,SIZE 2>&1
echo
echo "########## LVM: PV/VG/LV ##########"
pvs 2>&1
echo "---"
vgs 2>&1
echo "---"
lvs 2>&1
echo
echo "########## 关键挂载 (containerd/kubelet/var-log/mnt-data) ##########"
mount | grep -E "containerd|kubelet|/var/log|/mnt/data|dm-" 2>&1
echo
echo "########## /var/log/pods symlink ##########"
ls -la /var/log/pods 2>&1
echo "--- readlink ---"
readlink -f /var/log/pods 2>&1
echo "--- target exists? ---"
test -d "$(readlink /var/log/pods 2>/dev/null)" && echo "TARGET DIR EXISTS: yes" || echo "TARGET DIR EXISTS: no"
echo
echo "########## /var/lib/containerd/log/ ##########"
ls -la /var/lib/containerd/log/ 2>&1 || echo "NOT EXIST"
echo
echo "########## containerd.service.d drop-in ##########"
cat /etc/systemd/system/containerd.service.d/*.conf 2>&1
echo
echo "########## kubelet.service.d drop-in ##########"
cat /etc/systemd/system/kubelet.service.d/*.conf 2>&1
echo
echo "########## ti-disk-init.sh 挂载相关逻辑 ##########"
grep -nE "log/pods|/var/log|/var/lib/containerd|mountpoint|symlink|ln -s" /usr/local/bin/ti-disk-init.sh 2>&1 | head -30
