from graph_build import graph_

def main():
    graph = graph_
    inputs = {"question": "Advantages of RLHF"}
    result = graph.invoke(inputs)
    print(result)

if __name__ == "__main__":
    main()
