# エイリアスをスクリプト内で使えるようにする
# shopt -s expand_aliases
# source ~/.profile

# マウントするディレクトリの指定
repository_dir="$(pwd)"
data_dir="$(pwd)/../data"
image_dir="/data_server_storage/clients/cvlab/kc-augmentation-tool/20200304_485classes_100k/regi_legacy"
note_dir="$(pwd)/../note"

# run
docker run -it --rm \
--user 1043:1043 \
--runtime=nvidia \
-p 49194:8888 \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/shadow:/etc/shadow:ro \
-v /etc/sudoers.d:/etc/sudoers.d:ro \
-v /data_server_storage2/docker/setting/home:/home \
-e NVIDIA_VISIBLE_DEVICES=all \
-v $repository_dir:/working/rfcn3k \
-v $data_dir:/working/rfcn3k/data \
-v $image_dir:/working/rfcn3k/data/20200304_485classes_100k/regi_legacy \
-v $note_dir:/working/note \
-w /working \
syoukera/rfcn3k:multi-gpu_note \
jupyter notebook
