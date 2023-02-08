
#!/bin/bash

pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d matteobiglioli/msa-cnn
unzip msa-cnn.zip -d cache
rm msa-cnn.zip