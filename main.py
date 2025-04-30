from graph_build import graph

def main():
    graph = graph
    inputs = {"question": "What are agent memories?"}
    result = graph.invoke(inputs)
    print(result)

if __name__ == "__main__":
    main()
