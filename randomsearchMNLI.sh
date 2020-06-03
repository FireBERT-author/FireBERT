for i in {1..100}; do 
    python randomsearchMNLI.py 
    rm -r lightning_logs
done
