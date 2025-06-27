#!/bin/bash
MESSAGE="hello linux"
FILE_PATH="/tmp/hello.txt"
REMOTE_USER="guoxuanliu"
REMOTE_HOST="10.12.45.189"
REMOTE_PATH="/home/guoxuanliu/TestFolder"


error_exit(){
    echo "$1" >&2
    exit 1
}


echo "$MESSAGE">"$FILE_PATH"||error_exit"无法写入文件$FILE_PATH"


if [ ! -f "$FILE_PATH" ]; then
    error_exit "文件 $FILE_PATH 不存在！"
fi

echo "正在复制文件到远程服务器..."
scp -P 32222 "$FILE_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" || error_exit"复制失败！"

echo "操作完成！文件以复制到$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

