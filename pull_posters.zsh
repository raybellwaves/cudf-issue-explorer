#!/bin/zsh
#
# ./pull_posters.zsh

CSV_FILE="issue_posters.csv"
awk -F, 'NR>1 {print $2}' $CSV_FILE | while read -r username; do
    echo "Fetching data for user: $username"
    gh api users/$username >"poster_${username}_data.json"
done
