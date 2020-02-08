# エイリアスをスクリプト内で使えるようにする
# shopt -s expand_aliases
# source ~/.profile

# マウントするディレクトリの指定
repository_dir="$(pwd)"
data_dir="/home/a_shioyoke/data"

# run
docker run -it --rm \
--user 1043:1043 \
--runtime=nvidia \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/shadow:/etc/shadow:ro \
-v /etc/sudoers.d:/etc/sudoers.d:ro \
-v /data_server_storage2/docker/setting/home:/home \
-e NVIDIA_VISIBLE_DEVICES=all \
-v $repository_dir:/working/rfcn3k \
-v $data_dir:/working/data \
-w /working/rfcn3k \
syoukera/rfcn3k:multi-gpu \
bash