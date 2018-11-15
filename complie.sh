#! /bin/sh
version=4
case $1 in
	1|2|3|4|Zerobeta)
		version=$1;;
	*)
		echo 'usage: make version=n, n could be 1/2/3/4/Zerobeta, default is 4';;
esac

make "version=$version"
