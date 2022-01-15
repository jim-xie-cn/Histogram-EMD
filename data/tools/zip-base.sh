cd ../base
tar cvzf - ./*.csv | split --bytes=30MB - base.tar.gz.
rm -rf *.csv
