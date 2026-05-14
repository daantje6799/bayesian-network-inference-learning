"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""

import os
from pathlib import Path
from read_bayesnet import BayesNet
from VE import run_part1_bonus, run_part1_ve
from MAP import run_part2_map
from EM import run_part3_em

earthquake_bif  = Path('earthquake.bif')
dataset         = Path('endorisk_new.bif')
data_file       = Path('simulation_data_hid_names.dat')
learned_bif     = Path('learned_endorisk.bif')
results_path    = Path('log_endorisk.txt')

em_mode = os.getenv('EM_MODE', 'final')
sample_rows = int(os.getenv('EM_SAMPLE_ROWS', 500))
restarts = int(os.getenv('EM_RESTARTS', 5))
max_iter = int(os.getenv('EM_MAX_ITER', 10))
ve_heuristic = os.getenv('VE_HEURISTIC', 'min_fill')

def main():
    net = BayesNet(str(earthquake_bif))
    ve_sum,  _          = run_part1_ve(net, heuristic=ve_heuristic)
    ve_bonus            = run_part1_bonus(dataset, earthquake_bif)
    map_sum, map_log    = run_part2_map(net, {'Alarm': 'True'})
    em_sum,  em_log     = run_part3_em(dataset, data_file, learned_bif,
                                       sample_rows, restarts, max_iter)

    sections = [
        '=== Assignment Detailed Results ===',
        f'EM mode={em_mode} (restarts={restarts}, max_iter={max_iter})',
        f'VE heuristic={ve_heuristic}',
        '', '=== Part 1 (VE) ===', ve_sum,
        *((['', ve_bonus.rstrip()]) if ve_bonus else []),
        '', '=== Part 2 (MAP) ===',  (map_log or map_sum).rstrip(),
        '', '=== Part 3 (EM) ===',   (em_log  or em_sum).rstrip(),
        '', '=== Summary ===', ve_sum, map_sum, em_sum,
    ]
    results_path.write_text('\n'.join(sections) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()