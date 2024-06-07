import time

class Timer:
    def __init__(self):
        self.SECONDS_TO_MILLISECONDS = 1000
        self.MINUTES_TO_SECONDS = 60
        self.HOURS_TO_MINUTES = 60
        self.HOUR_TO_SECONDS = self.MINUTES_TO_SECONDS * self.HOURS_TO_MINUTES
        
        return
    
    def get_formatted_time(self):
        time_elapsed_in_seconds = self.get_elapsed_time()
        
        hours = int(time_elapsed_in_seconds // self.HOUR_TO_SECONDS)
        minutes = int((time_elapsed_in_seconds % self.HOUR_TO_SECONDS) // self.MINUTES_TO_SECONDS)
        seconds = int(time_elapsed_in_seconds % self.MINUTES_TO_SECONDS)
        milliseconds = int((time_elapsed_in_seconds - int(time_elapsed_in_seconds)) * self.SECONDS_TO_MILLISECONDS)
        
        time_string = ""
        if hours > 0:
            time_string += f"{hours} hour{'s' if hours > 1 else ''} "
            
        if minutes > 0:
            time_string += f"{minutes} minute{'s' if minutes > 1 else ''} "
            
        if seconds > 0:
            time_string += f"{seconds} second{'s' if seconds > 1 else ''} "
            
        if milliseconds > 0:
            time_string += f"{milliseconds} millisecond{'s' if milliseconds > 1 else ''}"
            
        return time_string
    
    def get_elapsed_time(self):
        return (self.end_time - self.start_time)
    
    def start(self):
        self.start_time = time.perf_counter()
        return
    
    def stop(self):
        self.end_time = time.perf_counter()
        return
    