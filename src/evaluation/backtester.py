class Backtester:
    def __init__(self, y_true, y_pred, dates, initial_capital: int = 10000):
        self.y_true = y_true
        self.y_pred = y_pred
        self.dates = dates
        self.initial_capital = initial_capital
        self.capital = self.initial_capital

    def generate_signals(self):
        self.signals = []
        for y in self.y_pred:
            if y > 0:
                self.signals.append(1)
            elif y < -0.0008:
                self.signals.append(-1)
            else:
                self.signals.append(0)

    def simulate_wallet(self):
        self.capital_history = [self.initial_capital]
        for i in range(len(self.y_true)):
            daily_return = self.signals[i] * self.y_true[i]
            new_capital = self.capital_history[-1] * (1 + daily_return)
            self.capital_history.append(new_capital)
        
        self.capital = self.capital_history[-1]
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.capital_history)
        plt.title('Capital History')
        plt.xlabel('Days')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.show()