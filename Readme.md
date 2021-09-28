

# Install Dependencies 

brew install pyenv
pyenv install 3.8.5
pyenv global 3.8.5
pyenv version
python3 -V

# Install Tensorflow
pip3 install --upgrade pip
pip3 install tensorflow
pip3 install tensorflowjs
pip3 install matplotlib

# Train Model
python3 BetterDataSet.py

