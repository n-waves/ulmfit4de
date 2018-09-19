#!/usr/bin/env bash
HOME="$(cd $(dirname $0); cd ..; pwd)"
DATA="$HOME/data"
WORK="$HOME/work"

BTW17="$DATA/btw17"

mkdir -p "$BTW17"


[ -d "$BTW17/jsons" ] || unzip "$DATA/recorded-tweets.zip" -d "$BTW17/jsons"
rm -rf "$BTW17/jsons/__MACOSX/"

if [ ! -f "$BTW17/text.json" ] ; then
    echo "Concat json files and filter out tweets"
    FILES=$(find "$BTW17/jsons/" -iname "chunk-*.json" | sort)

    for f in $FILES; do
     cat $f | jq -r '[.[]|{text: .text}]' >> "$BTW17/text-m.json" || echo "Error in $f"
    done
    cat "$BTW17/text-m.json" | jq  -s  'add' > "$BTW17/text.json"
fi


if [ ! -f "$BTW17/text.txt" ] ; then
    echo "Concat json files and filter out tweets"
    FILES=$(find "$BTW17/jsons/" -iname "chunk-*.json" | sort)

    for f in $FILES; do
     cat $f | jq -r '[.[] | "\(.text)"  | gsub("\n";"\t") ] |join("\n") | @text' >>  "$BTW17/text.txt" || echo "Error in $f"
    done
fi


#if [ ! -f "$BTW17/text.txt" ] ; then
#   echo "Convert to txt"
#
#   cat "$BTW17/text-m.json" | jq -r 'add | .text | @text' >> "$BTW17/text.txt"
#fi

# figure out how to work with such a large set
#if [ ! -f "$BTW17/text.csv" ] ; then
#   echo "Convert to csv"
#   cat "$BTW17/text-m.json" | jq -r 'add | .text | @csv' >> "$BTW17/text.csv"
#fi