#!/bin/bash -e

root_password="$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 16; echo)"
user_password="$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 16; echo)"

install -d "${ROOTFS_DIR}/etc/systemd/system/getty@tty1.service.d"
install -m 644 files/noclear.conf "${ROOTFS_DIR}/etc/systemd/system/getty@tty1.service.d/noclear.conf"
install -v -m 644 files/fstab "${ROOTFS_DIR}/etc/fstab"

cat <<EOF > /pi-gen/deploy/users
root:${root_password}
mlpi:${user_password}
EOF

on_chroot << EOF
if ! id -u pi >/dev/null 2>&1; then
	adduser --disabled-password --gecos "" mlpi
fi
echo "mlpi:mlpi4321" | chpasswd
echo "root:${root_password}" | chpasswd
EOF


