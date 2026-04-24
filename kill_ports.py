import psutil
import os
import signal

ports_to_kill = [5001, 5002, 8500]

for proc in psutil.process_iter(['pid', 'name']):
    try:
        conns = proc.connections(kind='inet')
        for conn in conns:
            if conn.laddr.port in ports_to_kill:
                print(f"Killing process {proc.pid} on port {conn.laddr.port}")
                os.kill(proc.pid, signal.SIGTERM)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
