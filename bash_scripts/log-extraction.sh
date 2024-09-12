
#$project = project directory

cd $project

# For git logs
git log \
  --pretty=format:'%n{%n  "commit": "%H",%n  "author": "%an <%ae>",%n  "date": "%ad",%n  "message": "%f":FILES:' \
  $@ | \
  perl -ne '
  BEGIN{print "["};
  if ($i = (/:FILES:[\n\r]*$/../^$/)) {
    if ($i == 1) {
      s/:FILES:[\n\r]*$//;
      $message = $_;
    } elsif ($i =~ /E0$/) {
      #$print_files->();
      print_files();
      @files = ();
    } elsif ($_ !~ /^$/) {
      chomp $_;
      push @files, $_;
    }
  } else { print; }
  END { print_files(1); }

  sub print_files {
    $last_line = shift;
    print $message;
    @files ?
      print qq(,\n  "files": [\n@{[join qq(,\n), map {qq(    "$_")} map {json_escape($_)} @files]}\n  ]\n})
      : print "\n}";
    $last_line ? print "]" : print @files ? "," : ",\n";
  };
  sub json_escape { $_ = shift; s/([\\"])/\\\1/g; return $_; }'

# For git stats
git log \
  --numstat \
  --format='%H' \
  $@ | \
  perl -lawne '
      if (defined $F[1]) {
          print qq#{"insertions": "$F[0]", "deletions": "$F[1]", "path": "$F[2]"},#
      } elsif (defined $F[0]) {
          print qq#],\n"$F[0]": [#
      };
      END{print qq#],#}' | \
  tail -n +2 | \
  perl -wpe 'BEGIN{print "{"}; END{print "}"}' | \
  tr '\n' ' ' | \
  perl -wpe 's#(]|}),\s*(]|})#$1$2#g' | \
  perl -wpe 's#,\s*?}$#}#'
