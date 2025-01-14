import random
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
from deap import creator, base, tools, algorithms, gp
import numpy as np
import operator

# Load data
data_path = 'BTCUSD_1h_Coinbase.csv'  # Veri dosyasının yolunu güncelle
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
data.set_index('datetime', inplace=True)
print("Data loaded successfully. First few rows:")
print(data.head())

# Updated grammar
grammar = {
    'Rule': ['{Condition}'],
    'Condition': ['{Comparison}', '({Condition} and {Condition})', '({Condition} or {Condition})', 'not ({Condition})'],
    'Comparison': ['{Indicator} {ComparisonOp} {Indicator}', '{Indicator} {ComparisonOp} {Number}'],
    'ComparisonOp': ['>', '<', '>=', '<=', '=='],
    'Indicator': ['self.sma2', 'self.sma3', 'self.sma5', 'self.sma10', 'self.roc3', 'self.roc12',
                  'self.high_3', 'self.low_3', 'self.data.Open', 'self.data.Close', 'self.data.High', 'self.data.Low'],
    'Action': ['self.buy()', 'self.sell()', 'pass'],
    'Number': ['0', '1', '2', '5', '10', '20', '50', '100']
}

# Mapping function from genotype to phenotype
def map_genome_to_phenotype(genome, start_symbol='Rule'):
    def expand(symbol, depth=0):
        print(f"{'  ' * depth}Expanding symbol: {symbol}")
        if depth > 10:
            print(f"{'  ' * depth}Max recursion depth reached for symbol: {symbol}")
            return "self.data.Close[-1] > self.data.Open[-1]"

        if symbol not in grammar:
            print(f"{'  ' * depth}Returning terminal symbol: {symbol}")
            return symbol

        choices = grammar[symbol]
        if not genome:
            print(f"{'  ' * depth}Genome exhausted while expanding ({symbol})")
            if symbol == 'Condition':
                # Doğrudan bir karşılaştırma üret
                indicator1 = random.choice(grammar['Indicator'])
                indicator2 = random.choice(grammar['Indicator'])
                comparison_op = random.choice(grammar['ComparisonOp'])
                print(f"{'  ' * depth}Randomly selected for Condition: {indicator1} {comparison_op} {indicator2}")
                return f"{indicator1} {comparison_op} {indicator2}"
            elif symbol == 'Comparison':
                # Karşılaştırma için de aynı şekilde
                indicator1 = random.choice(grammar['Indicator'])
                indicator2 = random.choice(grammar['Indicator'])
                comparison_op = random.choice(grammar['ComparisonOp'])
                print(f"{'  ' * depth}Randomly selected for Condition: {indicator1} {comparison_op} {indicator2}")
                return f"{indicator1} {comparison_op} {indicator2}"

            else:
                return expand(choices[0], depth + 1)

        choice_index = genome.pop(0) % len(choices)
        choice = choices[choice_index]

        expanded = []
        for part in choice.split():
            if part.startswith('{') and part.endswith('}'):
                expanded.append(expand(part[1:-1], depth + 1))
            else:
                expanded.append(part)

        result = ' '.join(expanded)
        print(f"{'  ' * depth}Returning expanded symbol: {result}")
        return result

    final_rule = expand(start_symbol)
    print(f"Final generated rule: {final_rule}")
    return final_rule

def generate_random_genome(length):
    return [random.randint(0, 255) for _ in range(length)]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(genome, mutation_rate):
    return [gene if random.random() > mutation_rate else random.randint(0, 255) for gene in genome]

def select_parents(population, fitness_scores):
    # Tournament selection
    tournament_size = 3
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Define the trading strategy
class DynamicStrategy(Strategy):
    def init(self):
        super().init()
        self.sma2 = self.I(talib.SMA, self.data.Close, timeperiod=2)
        self.sma3 = self.I(talib.SMA, self.data.Close, timeperiod=3)
        self.sma5 = self.I(talib.SMA, self.data.Close, timeperiod=5)
        self.sma10 = self.I(talib.SMA, self.data.Close, timeperiod=10)
        self.roc3 = self.I(talib.ROC, self.data.Close, timeperiod=3)
        self.roc12 = self.I(talib.ROC, self.data.Close, timeperiod=12)
        self.high_3 = self.I(talib.MAX, self.data.High, timeperiod=3)
        self.low_3 = self.I(talib.MIN, self.data.Low, timeperiod=3)
        self.rule = None  # Kuralı saklamak için bir değişken ekle

    def next(self):
        if self.rule is None:
            return  # Kural tanımlanmamışsa bir şey yapma

        # Göstergelerin ve verilerin son değerlerini al
        close = self.data.Close[-1]
        open_price = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        sma2 = self.sma2[-1]
        sma3 = self.sma3[-1]
        sma5 = self.sma5[-1]
        sma10 = self.sma10[-1]
        roc3 = self.roc3[-1]
        roc12 = self.roc12[-1]
        high_3 = self.high_3[-1]
        low_3 = self.low_3[-1]

        # Kuralı değerlendir
        try:
            # Kuraldaki göstergeleri ve veri noktalarını gerçek değerlerle değiştir
            rule = self.rule.replace("self.data.Close[-1]", str(close))
            rule = rule.replace("self.data.Open[-1]", str(open_price))
            rule = rule.replace("self.data.High[-1]", str(high))
            rule = rule.replace("self.data.Low[-1]", str(low))
            rule = rule.replace("self.sma2[-1]", str(sma2))
            rule = rule.replace("self.sma3[-1]", str(sma3))
            rule = rule.replace("self.sma5[-1]", str(sma5))
            rule = rule.replace("self.sma10[-1]", str(sma10))
            rule = rule.replace("self.roc3[-1]", str(roc3))
            rule = rule.replace("self.roc12[-1]", str(roc12))
            rule = rule.replace("self.high_3[-1]", str(high_3))
            rule = rule.replace("self.low_3[-1]", str(low_3))

            # 'not' operatörünü işle
            rule = rule.replace("not", "not ")

            # Kuralı değerlendir ve koşulu kontrol et
            condition = eval(rule)

            print(f"Rule evaluation result: {condition}")
            if condition:
                if not self.position:
                    self.buy()
            elif self.position:
                self.position.close()

        except Exception as e:
            print(f"Error evaluating rule: {self.rule}")
            print(f"Error message: {str(e)}")
            print("Current indicator values:")
            print(f"sma2: {self.sma2[-1]}")
            print(f"sma3: {self.sma3[-1]}")
            print(f"sma5: {self.sma5[-1]}")
            print(f"sma10: {self.sma10[-1]}")
            print(f"roc3: {self.roc3[-1]}")
            print(f"roc12: {self.roc12[-1]}")
            print(f"high_3: {self.high_3[-1]}")
            print(f"low_3: {self.low_3[-1]}")
            print(f"Open: {self.data.Open[-1]}")
            print(f"Close: {self.data.Close[-1]}")
            print(f"High: {self.data.High[-1]}")
            print(f"Low: {self.data.Low[-1]}")
            return 0

# Define the fitness function
def fitness_function(rule):
    # print(f"Evaluating rule: {rule}")
    class DynamicStrategyWrapper(DynamicStrategy):
        def init(self):
            super().init()
            self.rule = rule

    bt = Backtest(data, DynamicStrategyWrapper, cash=100000, commission=0.002)
    try:
        stats = bt.run()
        return stats['Equity Final [$]']
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        return 0

# Define the genetic algorithm
def genetic_algorithm(population_size, genome_length, generations, mutation_rate):
    population = [generate_random_genome(genome_length) for _ in range(population_size)]

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")
        fitness_scores = []
        for i, genome in enumerate(population):
            print(f"\nIndividual ({i+1}/{population_size})")
            print(f"Genome: {genome}")
            rule = map_genome_to_phenotype(genome.copy()) # Use a copy of the genome
            print(f"Generated rule: {rule}")
            fitness = fitness_function(rule)
            fitness_scores.append(fitness)
            print(f"Fitness score: {fitness}")

        # Select parents
        parents = select_parents(population, fitness_scores)

        # Create new population
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

        best_genome = max(zip(population, fitness_scores), key=lambda x: x[1])[0]
        best_strategy = map_genome_to_phenotype(best_genome.copy())
        best_fitness = max(fitness_scores)
        print(f"\nGeneration {generation + 1} Summary:")
        print(f"Best Strategy = {best_strategy}")
        print(f"Best Fitness = {best_fitness}")
        print(f"Average Fitness = {sum(fitness_scores) / len(fitness_scores)}")

    return population

# Run the Genetic Algorithm
population = genetic_algorithm(population_size=50, genome_length=50, generations=3, mutation_rate=0.01)

# Get the best strategy
best_genome = max(population, key=lambda genome: fitness_function(map_genome_to_phenotype(genome)))
best_strategy = map_genome_to_phenotype(best_genome)
print(f"\nBest Overall Strategy: {best_strategy}")

# En iyi stratejiyi bir dosyaya kaydet
with open("best_strategy.txt", "w") as f:
    f.write(best_strategy)