#!/usr/bin/env bash


val_annotations_filepath=${1:-"val_annotations.txt"}
src=${2:-"images/"}

echo "Preparing torch image folder dataset structure..."
while IFS=$'\t', read -r filename id rest
do 
	dst="./dataset/$id/$filename"
	echo -e "\e[1A\e[KCopying $filename to $dst"
	mkdir -p "./dataset/$id/" && cp "./$src/$filename" "$_"
done < $val_annotations_filepath
