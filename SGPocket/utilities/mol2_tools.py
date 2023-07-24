import datetime


def clean_mol2(mol2_in: str, mol2_out: str):
    """Clean MOL2 file (keep only ATOM)

    Args:
        mol2_in (str): Input MOL2 file
        mol2_out (str): Output MOL2 file
    """
    atom = False
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    new_lines_tab = []
    with open(mol2_in, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#	Modifying user name:'):
                tab_line = line.split(':')
                new_line = tab_line[0] + ': ' + 'SGPocket\n'
                new_lines_tab += [new_line]
            elif line.startswith('#	Modification time:'):
                tab_line = line.split(':')
                new_line = tab_line[0] + ': ' + now_str + '\n'
                new_lines_tab += [new_line]
            elif line.startswith('@<TRIPOS>ATOM'):
                atom = True
                new_lines_tab += ['@<TRIPOS>ATOM\n']
            elif line.startswith('@<TRIPOS>BOND'):
                atom = False
                new_lines_tab += ['@<TRIPOS>BOND\n']
            elif atom:
                new_line = line[:77]+' \n'
                new_lines_tab += [new_line]
            else:
                new_lines_tab += [line]
    with open(mol2_out, 'w') as fout:
        fout.writelines(new_lines_tab)
