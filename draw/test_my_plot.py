import sys

import read_from_csv as csv

import networkx
import matplotlib.pyplot


def main():
    matrix_str = csv.open_matrix('data.txt')

    if not csv.good_matrix(matrix_str):
        print('Bad matrix.')
        sys.exit(1)

    print(matrix_str)

    graph = create_graph(matrix_str)
    # print(graph.edges)
    draw_graph(graph)


    pass


def create_graph(matrix_str):
    graph = networkx.Graph()
    graph.add_nodes_from([i for i in range(len(matrix_str))])
    for index_line in range(len(matrix_str)):
        for index_number in range(len(matrix_str[index_line])):
            if matrix_str[index_line][index_number] == '1':
                graph.add_edge(index_line, index_number)
    return graph


def draw_graph(graph, path_to_save=None):
    networkx.draw(graph)
    if not path_to_save is None:
        matplotlib.pyplot.savefig(path_to_save)
    matplotlib.pyplot.show()
    pass



if __name__ == '__main__':
    main()
