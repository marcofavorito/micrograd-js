#!/usr/bin/env bash

# Run this script from the project root directory.

set -e

# build the library
yarn run build

# checkout gh-pages branch
rm -rf demo-build
mkdir demo-build
cd demo-build
git init 
git remote add origin git@github.com:marcofavorito/micrograd-js.git
git fetch origin gh-pages
git checkout gh-pages

# update files
rm -rf ./*
cp -r ../demo/* .
rm ./lib/microgradjs.js
cp ../dist/microgradjs.js ./lib


# commit changes
git add .
git commit -m "Update gh-pages."
git push origin gh-pages

set +e

# restore working dir
cd ..

