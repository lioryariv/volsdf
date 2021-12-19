confsmkdir -p data
cd data
echo "Downloading the DTU dataset ..."
wget https://www.dropbox.com/s/s6psnh1q91m4kgo/DTU.zip
echo "Start unzipping ..."
unzip DTU.zip
echo "DTU dataset is ready!"
rm -f DTU.zip
echo "Downloading the BlendedMVS dataset ..."
wget https://www.dropbox.com/s/c88216wzn9t6pj8/BlendedMVS.zip
echo "Start unzipping ..."
unzip BlendedMVS.zip
echo "BlendedMVS dataset is ready!"
rm -f BlendedMVS.zip