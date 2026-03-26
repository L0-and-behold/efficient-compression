# Create file with header first
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv > gpu_log.csv

# Then append data without headers
while true; do 
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> gpu_log.csv
    sleep 0.25
done