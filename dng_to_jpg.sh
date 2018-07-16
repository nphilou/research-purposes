#!/usr/bin/env bash

ufraw-batch *.dng --silent --out-type=jpeg --compression=95 --wb=camera --out-path=./jpg
