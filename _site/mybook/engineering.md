# git

遇到 server certificate verification failed. CAfile: /etc/ssl/certs/ca-certificates.crt CRLfile: none

```
git config --global http.sslverify false
```
git 退出nano界面：Ctrl + X然后输入y

## clone

clone branch

```
git clone -b breast https://.....
```

Git下放另一个git，在clone的时候：

```
git clone --recursive
```

## push更新

```
git pull origin breast
git add .
git commit -m 'message'
git push origin breast
```
## pull

```
git pull
```
如果更新不能merge，需要stash
```
git stash
```

## .gitignore

[A collection of useful .gitignore templates](https://github.com/github/gitignore)

# requirements.txt

```
pip install -r requirements.txt
```