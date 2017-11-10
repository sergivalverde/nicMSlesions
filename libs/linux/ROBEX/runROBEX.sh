#!/bin/bash


if [ $# -lt 2 ]
then
  echo "ROBEX requires at least 2 arguments"
  exit $E_BADARGS
fi

# Absolute path to output script
SCRIPT=$(readlink -f $0)
# Absolute path this script is in
SCRIPTPATH=`dirname $SCRIPT`
# Absolute path to input
INPUT=$(readlink -f $1)
# Absolute path to output1
OUTPUT1=$(readlink -f $2)
# Absolute path to output2 (if it is there)
if [ $# -gt 2 ]
then
	OUTPUT2=$(readlink -f $3)
else
	OUTPUT2=""
fi

cd $SCRIPTPATH
if [ $# -gt 3 ]
then
	cmd="./ROBEX $INPUT $OUTPUT1 $OUTPUT2 $4"
else
	cmd="./ROBEX $INPUT $OUTPUT1 $OUTPUT2"
fi

eval $cmd

exit

