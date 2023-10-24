可以參考官方的範例
https://github.com/buresu/ndi-python

注意！安裝環境有特別限制，請不要安裝python 3.10以上的版本！
- Windows x64 Python(3.7-3.10)
- macOS x64(>=10.12),arm64(>=11.0) Python(3.7-3.10)
- Linux x64,aarch64 Python(3.7-3.10)

記得install以下套件
```
pip install opencv-python
pip install ndi-python
pip install numpy
```

如果你是用Ubuntu Linux的話，還要安裝以下套件
```
sudo apt install avahi-daemon
sudo systemctl enable --now avahi-daemon
```