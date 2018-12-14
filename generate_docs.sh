#!/usr/bin/env bash
cd ../hickle/docs
make html
cd ../../hickle_docs

git checkout gh-pages
cp ../hickle/README.md ./
cp -r ../hickle/docs/build/html/* ./

git add .
git commit -m "Updated documenation: $(date)"
git push origin gh-pages

