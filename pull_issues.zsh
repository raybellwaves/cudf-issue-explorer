#!/bin/zsh
#
# ./pull_issues.zsh

REPO="rapidsai/cudf"
issue_numbers=($(gh issue list --repo $REPO --state open --limit 1000 --json number | jq -r '.[].number'))
for issue_number in "${issue_numbers[@]}"; do
    echo $issue_number
    padded_issue_number=$(printf "%05d" $issue_number)
    gh issue view --repo $REPO $issue_number --json assignees,author,body,closed,closedAt,comments,createdAt,id,labels,milestone,number,projectCards,projectItems,reactionGroups,state,title,updatedAt,url >issue_detail_$padded_issue_number.json
done
