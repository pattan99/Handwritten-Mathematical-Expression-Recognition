import datetime

class Logger:
    def __init__(self, file_name="logfile.log"):
        self.file_name = file_name
        # Xóa nội dung file khi tạo mới instance
        open(self.file_name, "w").close()
    
    def _write_log(self, level, message):
        with open(self.file_name, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] [{level}] {message}\n"
            f.write(log_message)
    
    def info(self, message):
        self._write_log("INFO", message)
    
    def warning(self, message):
        self._write_log("WARNING", message)
    
    def error(self, message):
        self._write_log("ERROR", message)