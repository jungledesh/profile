#!/bin/bash

git checkout main -f 
git pull origin main
git pull --rebase --autostash 


git checkout -b $1
git branch -D $2