class Backtester:
    def __init__(self, y_true, y_pred, dates, initial_capital: int = 10000):
        self.y_true = y_true
        self.y_pred = y_pred
        self.dates = dates
        self.initial_capital = initial_capital

    def generate_signals(self):
        print(self.y_pred)