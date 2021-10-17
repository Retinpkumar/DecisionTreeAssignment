from datetime import datetime

class Logger:
    """
    This class shall  be used for logging the details during app runtime.
    written by : Retin P Kumar
    """
    def __init__(self):
        # Instantiating variables
        self.now = datetime.now()
        self.time = self.now.strftime("%H:%M:%S")
        self.date = self.now.date()

    def log(self, file_object, log_message):
        # Recoding to file
        file_object.write("\n" + str(self.date) + "/" + str(self.time) + "\t" + log_message + "\n")
