be sure to run all command under this xmind folder: 
0. pip install -r requirements.txt
1. On ubuntu: find the xmind install path: whereis xmind, note this may not be real location of binary, could be the link. for exp, here is my real install path: /opt/Xmind/
2. go to the xmind path, backup the files:  sudo cp /opt/Xmind/resources/app.asar /opt/Xmind/resources/app.asar.bak
3. Now copy all documents from xmind install path to here: sudo cp -r /opt/Xmind/resources/* ./
4. Do crack: python xmind.py
5. copy back the generated app.asar: sudo cp -r app.asar /opt/Xmind/resources/

Now. relaunch the xmind, should be cracked..
