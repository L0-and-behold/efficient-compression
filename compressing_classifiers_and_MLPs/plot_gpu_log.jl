using CSV, DataFrames, Plots

df = CSV.read("gpu_log.csv", DataFrame)
plot(df.timestamp, df[!, "memory.used [MiB]"], 
     xlabel="Time", ylabel="VRAM (MiB)", 
     title="GPU Memory Usage", legend=false)