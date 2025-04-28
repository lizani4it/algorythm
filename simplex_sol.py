import sys
from fractions import Fraction
from typing import List

class SimplexSolver:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.table: List[List[Fraction]] = []
        self.basis: List[int] = []
        self.solutions: List[List[Fraction]] = []
        self.optimal_value = Fraction(0)
        self.status = ""
        self.phase = 1
        self.artificial_vars = 0
        self.original_cols = 0
        self.original_obj_func = []
        self.artificial_indices = []
        self.active_cols = []
        self.constant_z = Fraction(0)

    def read_from_file(self, filename: str) -> None:
        try:
            with open(filename, 'r') as file:
                line = file.readline().split()
                self.cols = int(line[0])
                self.rows = int(line[1])
                self.constant_z = Fraction(line[2]) if len(line) > 2 else Fraction(0)
                self.original_cols = self.cols
                obj_func = list(map(Fraction, file.readline().split()))
                if len(obj_func) != self.cols:
                    raise ValueError("Invalid objective function length")
                
                self.original_obj_func = obj_func
                self.table = []
                self.basis = []
                slack_var_index = self.cols
                self.artificial_indices = []
                
                temp_rows = []
                for i in range(self.rows):
                    row = list(map(Fraction, file.readline().split()))
                    if len(row) != self.cols + 1:
                        raise ValueError("Invalid constraint length")
                    temp_rows.append(row)
                
                used_columns = set()
                for i in range(self.rows):
                    coeffs = temp_rows[i][:self.cols]
                    b_i = temp_rows[i][self.cols]
                    
                    basis_var = -1
                    if b_i >= 0:
                        for j in range(self.cols):
                            if coeffs[j] == 1 and j not in used_columns:
                                is_basis = True
                                for k in range(self.rows):
                                    if k != i and temp_rows[k][j] != 0:
                                        is_basis = False
                                        break
                                if is_basis:
                                    basis_var = j
                                    used_columns.add(j)
                                    break
                    
                    if basis_var != -1:
                        coeffs += [Fraction(0)] * self.rows
                        coeffs.append(b_i)
                        self.basis.append(basis_var)
                    else:
                        coeffs += [Fraction(0)] * self.rows
                        coeffs[slack_var_index] = Fraction(1)
                        coeffs.append(b_i)
                        self.basis.append(slack_var_index)
                        self.artificial_indices.append(slack_var_index)
                        slack_var_index += 1
                    
                    self.table.append(coeffs)
                
                self.table.append([-x for x in obj_func] + [Fraction(0)] * (self.rows + 1))
                self.cols = slack_var_index
                self.artificial_vars = len(self.artificial_indices)
                self.active_cols = list(range(self.cols))
                
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found")
        except ValueError as e:
            raise ValueError(f"Invalid file format: {str(e)}")

    def add_artificial_variables(self) -> None:
        if not self.artificial_indices:
            return
        
        m_row = [Fraction(0)] * len(self.table[0])
        for i in range(self.rows):
            if self.basis[i] in self.artificial_indices:
                if self.table[i][-1] < 0:
                    self.table[i] = [-x for x in self.table[i]]
                for j in range(len(self.table[i])):
                    m_row[j] -= self.table[i][j]
        for idx in self.artificial_indices:
            m_row[idx] = Fraction(0)
        
        self.table.append(m_row)

    def remove_artificial_columns(self) -> None:
        for col_idx in sorted(self.artificial_indices, reverse=True):
            if col_idx in self.active_cols:
                self.active_cols.remove(col_idx)
                for i in range(len(self.table)):
                    self.table[i] = [self.table[i][j] for j in range(len(self.table[i])) if j in self.active_cols + [len(self.table[i]) - 1]]
                self.cols -= 1
        self.artificial_indices = []
        self.cols = len(self.active_cols)

    def get_pivot_col(self) -> int:
        target_row = -1
        min_val = Fraction(0)
        pivot_col = -1
        for j in self.active_cols:
            if self.phase == 1 and j >= self.original_cols:
                continue
            if self.table[target_row][j] < min_val:
                min_val = self.table[target_row][j]
                pivot_col = j
        if pivot_col == -1 and self.phase == 1:
            for j in self.active_cols:
                if self.table[target_row][j] < min_val:
                    min_val = self.table[target_row][j]
                    pivot_col = j
        return pivot_col

    def get_pivot_row(self, pivot_col: int) -> int:
        min_ratio = float('inf')
        pivot_row = -1
        for i in range(self.rows):
            if self.table[i][pivot_col] > 0 and self.table[i][-1] >= 0:
                ratio = self.table[i][-1] / self.table[i][pivot_col]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
        return pivot_row

    def pivot(self, pivot_row: int, pivot_col: int) -> None:
        pivot = self.table[pivot_row][pivot_col]
        if pivot == 0:
            raise ValueError("Pivot element is zero")
        
        for j in range(len(self.table[pivot_row])):
            self.table[pivot_row][j] /= pivot
        
        for i in range(len(self.table)):
            if i != pivot_row:
                factor = self.table[i][pivot_col]
                for j in range(len(self.table[i])):
                    self.table[i][j] -= factor * self.table[pivot_row][j]
        
        self.basis[pivot_row] = pivot_col

    def update_CO(self) -> None:
        if self.phase == 1:
            self.table[-1] = [Fraction(0)] * len(self.table[0])
            for i in range(self.rows):
                if self.basis[i] in self.artificial_indices:
                    for j in range(len(self.table[i])):
                        self.table[-1][j] -= self.table[i][j]
            for idx in self.artificial_indices:
                if idx in self.active_cols:
                    self.table[-1][idx] = Fraction(0)
        else:
            self.table[-1] = [Fraction(0)] * (len(self.active_cols) + 1)
            for j in range(len(self.active_cols)):
                c_j = self.original_obj_func[self.active_cols[j]] if self.active_cols[j] < self.original_cols else Fraction(0)
                for i in range(self.rows):
                    if self.basis[i] < self.original_cols:
                        self.table[-1][j] += self.original_obj_func[self.basis[i]] * self.table[i][j]
                self.table[-1][j] -= c_j
            z_value = Fraction(0)
            for i in range(self.rows):
                if self.basis[i] < self.original_cols:
                    z_value += self.original_obj_func[self.basis[i]] * self.table[i][-1]
            self.table[-1][-1] = z_value

    def check_optimal(self) -> bool:
        target_row = -1
        return all(self.table[target_row][j] >= 0 for j in range(len(self.active_cols)))

    def has_artificial_in_basis(self) -> bool:
        return any(b in self.artificial_indices for b in self.basis)

    def extract_solution(self) -> List[Fraction]:
        solution = [Fraction(0)] * self.original_cols
        for i in range(self.rows):
            if self.basis[i] < self.original_cols:
                solution[self.basis[i]] = self.table[i][-1]
        return solution

    def find_all_solutions(self) -> None:
        self.solutions = [self.extract_solution()]
        visited_bases = [tuple(self.basis)]
        
        non_basis = [j for j in self.active_cols if j not in self.basis]
        for col in non_basis:
            col_idx = self.active_cols.index(col)
            if col_idx < len(self.table[-1]) and self.table[-1][col_idx] == 0:
                pivot_row = self.get_pivot_row(col)
                if pivot_row != -1:
                    original_table = [row[:] for row in self.table]
                    original_basis = self.basis[:]
                    
                    self.pivot(pivot_row, col)
                    self.update_CO()
                    
                    new_basis = tuple(self.basis)
                    if new_basis not in visited_bases:
                        visited_bases.append(new_basis)
                        self.solutions.append(self.extract_solution())
                    
                    self.table = original_table
                    self.basis = original_basis

    def solve(self) -> None:
        self.add_artificial_variables()
        self.phase = 1
        
        while not (self.check_optimal() and self.table[-1][-1] == 0 and not self.has_artificial_in_basis()):
            pivot_col = self.get_pivot_col()
            if pivot_col == -1:
                if self.has_artificial_in_basis():
                    artificial_vars = [f"x{i+1}" for i in self.basis if i in self.artificial_indices]
                    self.status = f"Задача несовместна, так как осталась искусственная базисная переменная: {', '.join(artificial_vars)}"
                else:
                    self.status = "Система ограничений не совместна"
                return
            
            pivot_row = self.get_pivot_row(pivot_col)
            if pivot_row == -1:
                self.status = "Система ограничений не совместна"
                return
            
            self.pivot(pivot_row, pivot_col)
            self.update_CO()
        
        self.table.pop() if self.artificial_indices else None
        self.remove_artificial_columns()
        self.phase = 2
        self.update_CO()
        
        while not self.check_optimal():
            pivot_col = self.get_pivot_col()
            if pivot_col == -1:
                break
            
            pivot_row = self.get_pivot_row(pivot_col)
            if pivot_row == -1:
                self.status = "Целевая функция не ограничена"
                return
            
            self.pivot(pivot_row, pivot_col)
            self.update_CO()
        
        self.optimal_value = self.table[-1][-1]
        self.find_all_solutions()
        self.status = "Оптимальное решение найдено"
        if len(self.solutions) > 1:
            self.status += " (существует несколько оптимальных решений)"

    def print_solution(self) -> None:
        print("Результат решения:")
        print(self.status)
        
        if self.status.startswith("Оптимальное решение найдено"):
            if all(c == 0 for c in self.original_obj_func):
                print(f"Целевая функция константна (Z = {self.constant_z})")
            else:
                total_value = self.optimal_value + self.constant_z
                print(f"Оптимальное значение целевой функции: {total_value}")
                if len(self.solutions) > 1:
                    print("Найдено несколько базовых решений:")
                    for idx, sol in enumerate(self.solutions, 1):
                        sol_str = ", ".join(f"x{i+1} = {val}" for i, val in enumerate(sol[:self.original_cols]))
                        print(f"Решение {idx}: {sol_str}")
                else:
                    for i, val in enumerate(self.solutions[0][:self.original_cols], 1):
                        print(f"x{i} = {val}")
        
        print("\nФинальная симплекс-таблица:")
        headers = [f"x{j+1}" for j in self.active_cols] + ["b"]
        print("б.п. | " + " | ".join(headers))
        
        for i in range(self.rows):
            row_str = f"x{self.basis[i]+1} | " + " | ".join(str(self.table[i][j]) for j in range(len(self.active_cols))) + f" | {self.table[i][-1]}"
            print(row_str)
        
        if self.phase == 1:
            z_row_str = "M   | " + " | ".join(str(self.table[-1][j]) for j in range(len(self.active_cols))) + f" | {self.table[-1][-1]}"
        else:
            z_row_str = "Z   | " + " | ".join(str(self.table[-1][j]) for j in range(len(self.active_cols))) + f" | {self.table[-1][-1]}"
        
        print(z_row_str)

def main():
    solver = SimplexSolver()
    try:
        solver.read_from_file("input.txt")
        solver.solve()
        solver.print_solution()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
