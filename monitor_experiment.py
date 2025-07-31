#!/usr/bin/env python3
"""
Simple experiment monitor to check training progress.
"""

import time
import psutil
import os

def monitor_experiment():
    """Monitor the running experiment."""
    
    print("HepG2 VAE Experiment Monitor")
    print("="*40)
    
    # Check if Python processes are running
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and 'hepg2_simple_experiment.py' in ' '.join(proc.info['cmdline']):
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"Found {len(python_processes)} experiment process(es) running")
        for proc in python_processes:
            try:
                cpu_percent = proc.cpu_percent()
                memory_info = proc.memory_info()
                print(f"  PID: {proc.pid}")
                print(f"  CPU: {cpu_percent:.1f}%")
                print(f"  Memory: {memory_info.rss / 1024**2:.1f} MB")
            except:
                print(f"  PID: {proc.pid} (info unavailable)")
    else:
        print("No experiment processes currently running")
    
    # Check for recent log files or results
    experiments_dir = "experiments"
    if os.path.exists(experiments_dir):
        # Look for recent result files
        import glob
        result_files = glob.glob(os.path.join(experiments_dir, "hepg2_simple_results_*.json"))
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            file_time = os.path.getctime(latest_file)
            time_ago = time.time() - file_time
            
            if time_ago < 3600:  # Less than 1 hour ago
                print(f"\nLatest results: {os.path.basename(latest_file)}")
                print(f"Generated: {time_ago/60:.1f} minutes ago")
            else:
                print(f"\nLatest results: {os.path.basename(latest_file)}")
                print(f"Generated: {time_ago/3600:.1f} hours ago")
        else:
            print("\nNo result files found yet")
    
    # System resource info
    print(f"\nSystem Resources:")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"  Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"  Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")

if __name__ == '__main__':
    monitor_experiment()
