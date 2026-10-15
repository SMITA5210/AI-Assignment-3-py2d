# Getting Started 

This repository is a fork of the PY2D repository. It simplifies / abstracts out many aspects of the running process so you can get setup as soon as possible.
You will require `Python>=3.11` and either `x86_64` Linux (with `glibc>=2.29` -> Ubuntu 20.04+, Debian 11+) or macOS on Apple Silicon. The setup runs well with 8GB of RAM, though even 4GB should be sufficient.

```
git clone https://github.com/TheAbry/COL333-Assignment-3-py2d-fork.git py2d
cd py2d
```

You can use conda OR create a venv at this step, though it is recommended to use conda.
```
conda create -n ai-a3 python=3.11 -y && conda activate ai-a3
pip install -r requirements.txt
```

Next, we need to download the binaries for the server and the proxy. This step differs on Linux and Mac since the binaries are different.

### Linux
```
cd scripts
./download-rcssserver.sh
./download-proxy.sh
cd rcssserver && ./rcssserver --appimage-extract && cd ..
cd proxy
for b in sample_player sample_coach sample_trainer; do
    mkdir -p extract_$b && cd extract_$b
    ../$b --appimage-extract >/dev/null
    cd ..
    mv $b $b.AppImage
    ln -s extract_$b/squashfs-root/AppRun $b
done
cd ..
```
Note: We extract the AppImage binaries so that students don't need to have FUSE on their machines (even if you have FUSE, extract them since the script expects the AppImage to be extracted.)

### Mac
```
cd scripts
./download-binaries-macos.sh
```

Now all that is left is to generate gRPC stubs and see if everything runs fine.

```
cd ..
./generate.sh

ls service_pb2*
python -c "import service_pb2_grpc; print('ok')"
```
This should print `ok`.

Final check if everything is setup correctly:
```
python3 a3.py doctor
python3 a3.py match STARTER FULL --right full
```

The doctor command does a final check of every part of the pipeline and reports any errors. Then, the last command should run a match between the starter agent and the sample agent.
