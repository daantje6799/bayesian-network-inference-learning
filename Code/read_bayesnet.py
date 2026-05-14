"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Representation of a Bayesian network read in from a .bif file.

"""

import pandas as pd

class BayesNet():
    """
    This class represents a Bayesian network.
    It can read files in a .bif format (if the formatting is
    along the lines of http://www.bnlearn.com/bnrepository/)

    Uses pandas DataFrames for representing conditional probability tables
    """

    def __init__(self, filename):
        """
        Construct a bayesian network from a .bif file

        """
        # Use instance-level state so loading multiple networks in one run is safe.
        self.values = {}
        self.probabilities = {}
        self.parents = {}
        self.name = ''
        
        # Read file once and cache lines for efficiency
        with open(filename, 'r', encoding='utf-8') as file:
            self._lines = file.readlines()

        for line_number, line in enumerate(self._lines):
            if line.startswith('network'):
                self.name = ' '.join(line.split()[1:-1])
            elif line.startswith('variable'):
                self.parse_variable(line_number)
            elif line.startswith('probability'):
                self.parse_probability(line_number)

    def parse_probability(self, line_number):
        """Parse the probability distribution efficiently using cached lines."""
        line = self._lines[line_number]
        variable, parents = self.parse_parents(line)
        next_line = self._lines[line_number + 1].strip()

        # If a variable has no parents, its probabilities start with table
        if next_line.startswith('table'):
            comma_sep_probs = next_line.split('table')[1].split(';')[0].strip()
            probs = [float(p) for p in comma_sep_probs.split(',')]
            rows = [{variable: value, 'prob': p} for value, p in zip(self.values[variable], probs)]
            self.probabilities[variable] = pd.DataFrame(rows)
        else:
            # Build rows as list of dicts for efficiency
            rows = []
            for i in range(line_number + 1, len(self._lines)):
                line = self._lines[i]
                if '}' in line:
                    break
                
                # Get the values for the parents
                comma_sep_values = line.split('(')[1].split(')')[0]
                values = [v.strip() for v in comma_sep_values.split(',')]

                # Get the probabilities for the variable
                comma_sep_probs = line.split(')')[1].split(';')[0].strip()
                probs = [float(p) for p in comma_sep_probs.split(',')]

                # Create a row for each value combination
                for value, p in zip(self.values[variable], probs):
                    rows.append({variable: value, **{p_name: p_val for p_name, p_val in zip(parents, values)}, 'prob': p})

            self.probabilities[variable] = pd.DataFrame(rows)

    def parse_variable(self, line_number):
        """Parse the name of a variable and its possible values using cached lines."""
        variable = self._lines[line_number].split()[1]
        line = self._lines[line_number + 1]
        start = line.find('{') + 1
        end = line.find('}')
        values = [value.strip() for value in line[start:end].split(',')]
        self.values[variable] = values

    def parse_parents(self, line):
        """
        Find out what variables are the parents
        Returns the variable and its parents
        """
        start = line.find('(') + 1
        end = line.find(')')
        variables = line[start:end].strip().split('|')
        variable = variables[0].strip()
        if len(variables) > 1:
            parents = variables[1]
            self.parents[variable] = [v.strip() for v in parents.split(',')]
        else:
            self.parents[variable] = []
        return variable, self.parents[variable]

    @property
    def nodes(self):
        """Returns the names of the variables in the network"""
        return list(self.values.keys())
