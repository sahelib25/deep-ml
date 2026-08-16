def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    elements_col = len(matrix[0])
    elements_row = len(matrix)

    means = []

    if mode == 'row':
        for row in range(elements_row):
            total = 0
            for col in range(elements_col):
                total += matrix[row][col]

            means.append(total / elements_col)

    elif mode == 'column':
        for col in range(elements_col):
            total = 0
            for row in range(elements_row):
                total += matrix[row][col]

            means.append(total / elements_row)

    return means