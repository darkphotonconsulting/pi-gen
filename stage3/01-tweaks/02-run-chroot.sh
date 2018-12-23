
echo "Updating Autologin Service Configuration"
/bin/sed -i 's#pi#mlpi#g' /etc/systemd/system/autologin@.service 
echo "Reloading SystemD"
/bin/systemctl daemon-reload 
echo "Restarting Autologin@TTY1"
/bin/systemctl restart autologin@tty1



