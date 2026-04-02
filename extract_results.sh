#!/bin/bash
# Usage: ./extract_results.sh <log_file>
# Extracts key metrics from each epoch

LOG=$1
echo "=== $(basename $LOG .log) ==="
echo ""

# Get all eval lines (contain wikitext_kl)
grep "'wikitext_kl'" "$LOG" 2>/dev/null | while IFS= read -r line; do
    epoch=$(echo "$line" | grep -oP "'epoch': '[^']*'" | cut -d"'" -f4)
    if [ -z "$epoch" ]; then
        epoch=$(echo "$line" | grep -oP "'epoch': [0-9]*" | cut -d' ' -f2)
    fi
    wiki_kl=$(echo "$line" | grep -oP "'wikitext_kl': '[^']*'" | cut -d"'" -f4)
    if [ -z "$wiki_kl" ]; then
        wiki_kl=$(echo "$line" | grep -oP "'wikitext_kl': [0-9.e-]*" | cut -d' ' -f2)
    fi
    broken=$(echo "$line" | grep -oP "'wikitext_broken': [A-Za-z]*" | cut -d' ' -f2)
    recall=$(echo "$line" | grep -oP "'recall_prob': '[^']*'" | cut -d"'" -f4)
    if [ -z "$recall" ]; then
        recall=$(echo "$line" | grep -oP "'recall_prob': [0-9.e-]*" | cut -d' ' -f2)
    fi
    acc_t0=$(echo "$line" | grep -oP "'forget_acc_t0': '[^']*'" | cut -d"'" -f4)
    if [ -z "$acc_t0" ]; then
        acc_t0=$(echo "$line" | grep -oP "'forget_acc_t0': [0-9.]*" | cut -d' ' -f2)
    fi
    acc_t1=$(echo "$line" | grep -oP "'forget_acc_t1': '[^']*'" | cut -d"'" -f4)
    if [ -z "$acc_t1" ]; then
        acc_t1=$(echo "$line" | grep -oP "'forget_acc_t1': [0-9.]*" | cut -d' ' -f2)
    fi
    retain_kl=$(echo "$line" | grep -oP "'retain_eval_kl': '[^']*'" | cut -d"'" -f4)
    if [ -z "$retain_kl" ]; then
        retain_kl=$(echo "$line" | grep -oP "'retain_eval_kl': [0-9.e-]*" | cut -d' ' -f2)
    fi
    printf "E%-3s | wiki_kl=%-8s broken=%-5s | recall=%-8s acc_t0=%-6s acc_t1=%-6s | ret_kl=%-8s\n" \
        "$epoch" "$wiki_kl" "$broken" "$recall" "$acc_t0" "$acc_t1" "$retain_kl"
done

# Robustness
rob=$(grep "Robustness:" "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
if [ -n "$rob" ]; then
    echo ""
    echo "Robustness: $rob"
fi
echo ""
