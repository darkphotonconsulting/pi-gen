
echo "Updating /etc/dphys-swapfile"

cat << EOF > /etc/dphys-swapfile
# where we want the swapfile to be, this is the default
#CONF_SWAPFILE=/var/swap

# set size to absolute value, leaving empty (default) then uses computed value
#   you most likely don't want this, unless you have an special disk situation
CONF_SWAPSIZE=1024

# set size to computed value, this times RAM size, dynamically adapts,
#   guarantees that there is enough swap without wasting disk space on excess
#CONF_SWAPFACTOR=2

# restrict size (computed and absolute!) to maximally this limit
#   can be set to empty for no limit, but beware of filled partitions!
#   this is/was a (outdated?) 32bit kernel limit (in MBytes), do not overrun it
#   but is also sensible on 64bit to prevent filling /var or even / partition
#CONF_MAXSWAP=2048
EOF

echo "Reload dphys-swapfile service"
/etc/init.d/dphys-swapfile stop
/etc/init.d/dphys-swapfile start

cd /root

/usr/bin/pip3 install mxnet-1.5.0-py2.py3-none-any.whl
