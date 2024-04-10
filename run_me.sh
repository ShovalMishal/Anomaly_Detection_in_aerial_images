sed 's/nameserver 10.*/nameserver 132.66.150.2/' /etc/resolv.conf >/tmp/R$$
cp /tmp/R$$ /etc/resolv.conf
python ./FullOODPipeline.py -c ./config.py