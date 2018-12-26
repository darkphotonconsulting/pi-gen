#!/bin/bash -e

rm -f "${ROOTFS_DIR}/etc/systemd/system/dhcpcd.service.d/wait.conf"


echo "Enable WPA Supplicant service"
on_chroot << EOF
systemctl enable wpa_supplicant.service
EOF
