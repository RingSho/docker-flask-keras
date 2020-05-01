# 機械学習マイクロサービス(Reastful-API, keras, tensorflow) on GPU Docker

## 開発環境

Ubuntu 18.04  
Docker 19.03.8  
nvidia-container-runtime 1.0.0-rc10  

## 前提

nvidia-smiがローカルで動くこと  
pythonでgpuを使った学習が行えていること  

## Docker

<https://docs.docker.com/engine/install/ubuntu/>

## Nvidia-container-runtime

<https://github.com/NVIDIA/nvidia-container-runtime>

## no cache build

`docker-compose build --no-cache`

## apt-get updateでハッシュ不適合のエラーが出たら

ホスト側で以下を実行  

`sudo rm -rf /var/lib/apt/lists/*`  
`sudo rm -rf /var/lib/apt/lists`  
`sudo apt-get update`

## コンテナ全stop

`docker stop $(docker ps -q)`

## コンテナ全削除

`docker ps -aq | xargs docker rm`

## イメージ全削除

`docker images -aq | xargs docker rmi`
