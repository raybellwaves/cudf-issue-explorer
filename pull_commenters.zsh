#!/bin/zsh
#
# ./pull_commenters.zsh

CSV_FILE="issue_commenters.csv"
awk -F, 'NR>1 {print $2}' $CSV_FILE | while read -r username; do
    echo "Fetching data for user: $username"
    gh api users/$username >"commenter_${username}_data.json"
done
