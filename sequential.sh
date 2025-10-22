#!/bin/bash
args_file=$1

if [[ $# -eq 0 ]] ; then
    echo 'no file argument given'
    exit 0
fi

MAX_TIME=810000
IFS=$'\n'
m=$(cat $args_file | wc -l) # number of arguments in args.txt file

for ((i=0; i<=$m; i++)); do
  line=$(sed -n "$((i+1))p" $args_file)
  if [[ "$line" != +(*"&"*|*"#"*) ]]; then
    cmd="python -u train.py $line"
    echo -e "\n\n\n"
    echo $cmd
    eval $cmd 
  fi
done
wait
