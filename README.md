# Docker

## no cache build

`docker-compose build --no-cache`

## コンテナ全stop

`docker stop $(docker ps -q)`

## コンテナ全削除

`docker ps -aq | xargs docker rm`

## イメージ全削除

`docker images -aq | xargs docker rmi`

## apt-get updateでハッシュ不適合のエラーが出たら

`sudo rm -rf /var/lib/apt/lists/*`
`sudo apt-get update`
