#!/usr/bin/env bash
git checkout gh-pages
cp ../hickle/README.md ./
cp -r ../hickle/docs/build/html/* ./

git add .
git commit -m "Updated documenation: $(date)"


