cd ../base
tar cvzf - ./*.csv | split --bytes=45MB - base.tar.gz.
rm -rf *.csv
