#!/usr/bin/env bash

for file in *.jpg;
		do
      echo $file;
      echo "(python-cubism 0 "$file" 10 2.5 0)";
      gimp -i -b "(python-cubism 0 \"$file\" 10 2.5 0)" -b '(gimp-quit 0)';
		done
